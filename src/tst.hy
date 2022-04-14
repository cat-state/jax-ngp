(import multiprocessing
        webdataset [WebDataset WebLoader]
        torchvision [transforms]
        jax
        jax.tree-util [tree-map Partial :as partial]
        jax.numpy :as np
        optax
        jax.example-libraries [stax]
        tqdm [tqdm trange]
        cv2
        numpy :as onp
        matplotlib [pyplot :as plt]
        pickle)

(import ngp)
(import hash)

(require hyrule *)

(defn coords [h w]
  (-> (np.meshgrid (np.linspace 0 1 h)
                   (np.linspace 0 1 w)
                   :indexing "ij")
      (np.stack :axis -1)
      (.reshape [-1 2])))

(defmacro "#dbg" [form]
  `(let [x ~form]
     (do
       (print ~(hy.repr form) x)
       x)))

(setv ortho-init (jax.nn.initializers.glorot-normal))
(defn linear [out-dim]
  (defn init-fn [rng input-shape]
    (setv [#* hd in-dim] input-shape)
    (,(, #* hd out-dim)
      (dict :W (ortho-init rng [in-dim out-dim]))))
  (defn apply-fn [params x #** kwargs]
    (@ x (:W params)))
  [init-fn apply-fn])


(defn grid-hash-feature-encoding [levels T]
  (stax.serial
   (stax.FanOut (- levels 4))
   (stax.parallel
     (ngp.GridEncodedFeatures T (* 2 4) 4)
     #* (lfor level (range 5 levels)
                    (ngp.encoding T 2 level :hasher hash.ngp-hash2)))
   (stax.FanInConcat -1)))



(defn hash-feature-encoding [levels T]
  (stax.serial
   (stax.FanOut (+ 0 levels))
   (stax.parallel
     #* (lfor level (range 0 levels)
                    (ngp.encoding T 2 level :hasher hash.ngp-hash2)))
   (stax.FanInConcat -1)))

(defn Mlp [out-dim [levels 12] [T (** 2 12)]]
   (stax.serial
     (hash-feature-encoding levels T)
     (linear 64) stax.Relu
     (linear 64) stax.Relu
     (linear out-dim) (stax.elementwise #%(+ %1 0.5))))

(defn imshow [x]
  (cv2.imshow "imshow" (-> (onp.asarray x) (ncut : : ::-1)))
  (cv2.waitKey 1))

(defn apply-optimizer [optimizer opt-state grad params]

  (setv [updates opt-state] (.update optimizer grad opt-state params))
  [opt-state
   (optax.apply-updates params updates)])

(defn mse [x y]
  (.mean (** (- x y) 2)))

(defn relative-mse [x y]
  (.mean (/ (** (- x y) 2) 
         (+ (** x 2) 0.01))))

(defn loss [model params x y]
  (psnr (model params x) y))

(defn train-step [apply-optimizer apply-model opt-state params x y]
  (setv [loss-val grad] ((jax.value-and-grad loss :argnums 1)
                         apply-model params x y))
  [loss-val
   (apply-optimizer opt-state grad params)])

#@(jax.jit 
    (defn sample-img [img key n-points]
      (setv [h w c] img.shape
             xy (jax.random.uniform key [n-points 2]
                    :minval (np.array [0 0])
                    :maxval (np.array [h w])))

      [(/ xy (np.array [h w]))
       (ngp.interpolate xy img)]))


(defn psnr [x y]
  (* 10.0 (/ (np.log (mse x y))
             (np.log 10.0))))

(defn get-1 [x]
  (get x 0))

(defn main []
  (setv batch-size 1)
  (setv URL "/home/a/openimages/openimages-train-000000.tar"
        cc12m "cc12m_00000.tar")

  (setv dataset (-> (WebDataset cc12m)
                  (.decode "rgb")
                  (.to-tuple "jpg")
                  (.map get-1))
      loader (WebLoader dataset :num-workers 0 :collate-fn list
                        :batch-size batch-size))


  (do

    (setv [init-weights mlp] (Mlp)
          [out-shape weights] (init-weights ngp.KEY (, 2))
          [key _] (jax.random.split ngp.KEY 2)
          adamw   (optax.adamw
                    (optax.exponential-decay :init-value 1e-2 
                                             :transition-steps 500 
                                             :decay-rate 0.75
                                             :transition-begin 1000
                                             :staircase True)
                    :b1 0.9 :b2 0.99 :eps 1e-15
                    :weight-decay 1e-6
                    :mask (tree-map (fn [t] (not-in "hash" t)) weights
                                      :is-leaf (fn [n] (in "hash" n))))
          adam (optax.adam (optax.exponential-decay :init-value 1e-2 
                                             :transition-steps 500 
                                             :decay-rate 0.75
                                             :transition-begin 500
                                             :staircase True) :b1 0.9 :b2 0.99 :eps 1e-15)
          optimizer (optax.chain adam)
          s-mlp (jax.jit mlp)
          steppy (partial train-step
                      (partial apply-optimizer optimizer)
                      (jax.vmap mlp [0 0]))
          jit-train (jax.profiler.annotate-function (jax.jit steppy) :name "train-step"))

   
   (mlp weights (coords 3 3))

   (print (np.size (get (jax.flatten-util.ravel-pytree weights) 0)))
   (setv il (tqdm (iter loader) :total (// 10000 batch-size)))
   (setv sizes [])

   (for [[img-idx imgs] (enumerate il)]
     (setv [h w c] (. (get imgs 0) shape))
     (if (< (min h w) 256)
       (continue))
     (.append sizes [h w])

    (tqdm.write f"img {img-idx} {h} {w}")
    (setv 
        [out-shape weights] (init-weights ngp.KEY (, 2))
        ensemble-weights (tree-map #%(np.stack (* batch-size [%1])) weights)
        opt-state (.init optimizer ensemble-weights))
    (setv imgs (tree-map #%(np.array %1) imgs))
    (for [i (range 0 1000)]
     (setv 
       [key #* sample-keys] (jax.random.split key (+ 1 (len imgs)))
       samples (lfor [img rng] (zip imgs sample-keys)
                     (sample-img img rng (* 256 256))))

     (setv
       [xys gt] (zip #* samples)
       [xys gt] [(np.stack xys)
                 (np.stack gt)])


     (setv  [loss-val [opt-state ensemble-weights]] 
           (jit-train opt-state ensemble-weights xys gt))

     (when (and (> i 0) (= 0 (% i 999))) 
      (tqdm.write (str loss-val))
      (imshow (-> (s-mlp (tree-map #%(get %1 0) ensemble-weights)
                     (coords 1024 1024)) 
                  (.reshape [1024 1024 3])))))

    (setv padded (.zfill (str img-idx) 5))
    (with [fp (open f"./latents_cc12m_214_gh/{padded}.pickle" "wb")]
      (pickle.dump ensemble-weights fp)))
      ;(break)
      

   (print (max (lfor [h w] sizes h))
          (max (lfor [h w] sizes w)))
   (print (.mean (np.array sizes))
          (.std (np.array sizes)))
   (cv2.destroyAllWindows)
   (setv [h w c] (. (get imgs 0) shape))
   (plt.imshow (.reshape (mlp (tree-map #%(get %1 0) ensemble-weights)
                             (coords h w)) [h w c]))
   (plt.show))) 

   
    
            
   
(when (= __name__ "__console__")
 (main)) 
        
(when (= __name__ "__main__")
  (multiprocessing.set-start-method "spawn")
  (multiprocessing.set-executable (. hy sys-executable))
  (main))
  ;(input)
  
