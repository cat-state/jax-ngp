(import numpy :as onp
        jax
        jax.experimental [sparse]
        jax.numpy :as np
        jax.tree-util [tree-map]
        jax.example-libraries [stax]
        jax.flatten-util [ravel-pytree]
        umap [UMAP plot]
        webdataset [WebDataset WebLoader]
        pickle
        matplotlib [pyplot :as plt]
        sklearn.decomposition [PCA]
        einops [rearrange]
        tqdm [tqdm trange]
        toolz [curry]
        ngp
        hash [ngp-hash2 R2-hash])
(require hyrule [-> ncut "#%"]
         util ["#dbg"])

(setv tree-map (curry tree-map))

(setv KEY (jax.random.PRNGKey 2022))

(setv bw 1)

(defn GroupNorm [num-groups [eps 1e-5]]
  (defn init-fn [rng input-shape]
    (setv [#* hd channels] input-shape)
    (when (!= 0 (% channels num-groups))
      (raise 
        (ValueError f"num-groups {num-groups} doesn't evenly divide {channels}")))
    (, input-shape []))
  (defn apply-fn [params x #** etc]
    (setv [#* hd channels] x.shape
          grouped (.reshape x [#* hd -1 num-groups]))
    (.reshape (jax.nn.normalize grouped :epsilon eps) 
      [#* hd channels]))
  [init-fn apply-fn])


(defn Get [i]
  (defn init-fn [rng input-shapes]
    (, (get input-shapes i) []))
  (defn apply-fn [_ x #** etc]
    (get x i))
  [init-fn apply-fn])



(defn residual+ [#* inner]
  (stax.serial
    (stax.FanOut 2)
    (stax.parallel
      ; Extract X from [time X] 
      (Get 1)
      (stax.serial #* inner))
    stax.FanInSum))



(setv SiLU (stax.elementwise jax.nn.silu))

(defn Linear₀ [out-dim]
  (stax.Dense out-dim :W-init jax.nn.initializers.zeros))

(defn ResMlp [lin-dim [mode "train"]] 
  (residual+
    (stax.parallel

      ; Time branch
      (stax.serial
        SiLU
        (stax.Dense lin-dim))

      ; X branch
      (stax.serial
        (GroupNorm 32)
        SiLU
        (stax.Dense lin-dim)))

    stax.FanInSum
    (GroupNorm 32)
    SiLU
    (stax.Dropout 0.2 mode)
    (Linear₀ lin-dim)))



(defn UNet [#* inner]

  (setv [inits applys] (zip #* inner))
  
  (assert (= 0 (% (len inner) 2)))
  
  (defn init-fn [rng input-shapes]
    (setv [time-shape x-shape] input-shapes
          params [])
    (for [[init rng] (zip inits (jax.random.split rng (len inits)))]

      (setv [x-shape param] (init rng [time-shape x-shape]))
      (.append params param))
    (, x-shape params))
  
  (defn apply-fn [params x rng]
    (assert (= (len params) (len applys))) 
    
    (setv [time x] x
          rngs (jax.random.split rng (len params))
          outputs [x])
    
    (for [[i [param rng apply-fn]] (enumerate (zip params rngs applys))]
      
      ; For the 2nd half of the layers, make a residual
      ; connection from the 1st half, so the last layer is connected
      ; to the first, the 2nd last to the 2nd, etc...
      
      (when (> i (// (len applys) 2))
        (setv x (+ x (get outputs (- (+ i 1))))))
      (setv x (apply-fn param [time x] :rng rng))
      (.append outputs x))
    x)

  [init-fn apply-fn])

(* 16 16)
(* 64 64)

(defn DDPM-Mlp [time-embed-dim lin-dim 
                [mode "train"]]
  (stax.serial
    (stax.parallel
      (stax.Dense time-embed-dim) (stax.Dense lin-dim))
    (UNet
      #* (lfor _ (range 4)
            (ResMlp lin-dim mode)))
    (GroupNorm 32)
    SiLU
    (Linear₀ lin-dim)))

(setv [init apply] (DDPM-Mlp 128 4096)
      [o-shape w] (init KEY [(, 1) (, 4096)])
      aj (jax.jit apply))
(import timeit [timeit])
(aj w [(np.ones 1) (np.ones 4096)] :rng KEY)
(print (timeit #%(.block-until-ready (aj w [(np.ones 1) (np.ones 4096)] :rng KEY)) :number 100))

(setv T (** 2 12))

(defn conv-offsets [bw]
  (.reshape 
    (np.dstack (np.meshgrid
                (np.arange (- bw) (+ 1 bw))
                (np.arange (- bw) (+ 1 bw))))
    [-1 2]))


(defn connectivity [points sample-offset hasher n]
  ; [num-samples]
  (setv points (rearrange (np.floor points) "n c -> n 1 c")
        h (hasher (+ points sample-offset) n)
        h0 (hasher points n)
        out (np.zeros [n n]))
  (.set (ncut out.at h0 h) 1.0))
(setv kernel (connectivity
               (jax.random.uniform KEY [(** 2 16) 2] 
                           :minval 0
                           :maxval sampling-rate) 
               (conv-offsets 3)
               hasher.ngp-hash2 T)
      ksparse (sparse.BCOO.fromdense kernel))


(defmacro dont [#* forms])

(import timeit [timeit])
(dont
  (plt.plot (. (.fit (PCA) kernel) explained-variance-ratio-))
  (plt.show))


(import optax
        jax.tree-util [Partial :as partial])
(import tst [train-step apply-optimizer Mlp sample-img])
(import hash :as hasher)
(ngp.grid-resolution 8 1.4 16)

(setv sampling-rate (ngp.grid-resolution 8 1.4 16))

(setv [init-weights hash-get] (ngp.HashEncodedFeatures T 3 8 hasher.iq-hash) 
      hash-get (jax.jit hash-get))
        

(defn fit [key img sampling-rate]
  (setv [out-shape weights] (init-weights key (, 2))
        adam (optax.adam 1e-2 :b1 0.9 :b2 0.99 :eps 1e-15)
        opt-state (.init adam weights)
        step (jax.jit (partial train-step
                       (partial apply-optimizer adam)
                       hash-get)))
  (setv weights (tree-map #%(np.zeros-like %1) weights))
  (for [i (trange 1000)]

    (setv [key] (jax.random.split key 1)
          gt-xy (jax.random.uniform key [(* 256 256) 2])
          gt (img gt-xy)
          [loss [opt-state weights]] (step opt-state weights gt-xy gt))
    (tqdm.write (str loss))
    (tst.imshow (.reshape (hash-get weights (tst.coords 256 256)) [256 256 3])))
    
  (import cv2)
  (cv2.destroyAllWindows)
  [weights hash-get])

(import importlib [reload])
(import tst)
(reload tst)
(defn ->01 [x]
  (+ 0.5 (* 0.5 x)))

(defn ->-11 [x]
  (- (* 2 x) 1))


(defn sym-lap [k]
  (setv deg (np.diag (np.sum k :axis 1))
        L (- deg k)
        [deg-inv-sqrt _] (optax.matrix-inverse-pth-root deg 2))
  (@ deg-inv-sqrt L deg-inv-sqrt))



(defn delta-fn [radius]
 (fn [xy] (np.where (< (np.linalg.norm (->-11 xy) :axis -1 :keepdims True) radius)
                   (np.ones [#* (ncut xy.shape :-1) 3]) 
                   (np.zeros [#* (ncut xy.shape :-1) 3])))) 

(defn sinc [radius]
  (fn [xy] (.repeat (np.prod (np.sinc (* (/ 1.0 radius) (->-11 xy))) :axis -1 :keepdims True) 3 -1)))

(do
 (plt.imshow
  ((delta-fn 0.5) (.reshape (tst.coords 1024 1024) [1024 1024 2])))
 (plt.show))

(do
 (plt.imshow
  ((sinc 0.5) (np.dstack (np.meshgrid (np.linspace 0 1 1024)
                                  (np.linspace 0 1 1024))))
  :cmap "RdBu")
 (plt.show))
 

(do
  (setv [delta-weights _] (fit KEY (delta-fn 0.05) sampling-rate)
        [sinc-weights _] (fit KEY (sinc 0.05) sampling-rate)))

(defn fit-kernel [x y ksize]
  (setv K (np.zeros [ksize ksize])
        adamw (optax.adamw 1e-4)
        opt-state (.init adamw K)
        step (jax.jit
               (partial train-step
                (partial apply-optimizer adamw)
                (partial #%(@ %2 %1)))))
  (for [i (range 100)]
    (setv [loss [opt-state K]] (step opt-state K x y)))
  (print loss)
  K)

(defn coords [sampling-rate]
  (* sampling-rate (tst.coords (int sampling-rate) (int sampling-rate))))

(. (:hash delta-weights) shape)
(setv dotc (jax.vmap #%(@ %1 %2 %2 %2 %2) [-1 None] :out-axes -1))

(setv sw {"hash" (dotc (:hash delta-weights) (sym-lap kernel))})
(. (:hash delta-weights) shape)

(do
  (setv res (int (* 1 sampling-rate)))
  (plt.imshow (.reshape (hash-get sw (tst.coords res res)) [res res 3]))
  (plt.show))

(.max (hash-get delta-weights (tst.coords res res)))

  ;(setv [skern resid rank s] (np.linalg.lstsq (. (:hash last-delta) T) (. (:hash last-sinc) T)))
(setv skern (fit-kernel (. last-delta T) (. last-sinc T) 4096))
(print skern.shape)
(plt.imshow skern)
(plt.show)
(plt.imshow (.fit-transform (PCA) skern))
(plt.show)
(setv [eigval eigvec] (np.linalg.eigh skern))
(plt.plot (ncut eigvec : :10)) (plt.show)
  (plt.imshow eigvec) (plt.show)
  

(do
  (plt.plot (ncut (get (np.linalg.eigh kernel) 1) : 10))
  (plt.show))
(do
  (plt.imshow (sym-lap kernel))
  (plt.show))
(defn matsq [x v]
  (@ v x v))
(setv sm (jax.jit (sparse.sparsify matsq))
      v (np.ones (** 2 12))
      jsq (jax.jit matsq))
(print 
  "mem"
  (.format "{:1f}%" (* 100 (/ ksparse.nse (** (len v) 2))))
  "sparse"
  (/ (timeit #%(.block-until-ready (sm ksparse v)) :number 1000) 1000)
  "dense"
  (/ (timeit #%(.block-until-ready (jsq kernel v)) :number 1000) 1000))

(setv dataset (-> (WebDataset "./latents_cc12m_gh.tar")
                  (.map-dict :pickle pickle.loads)
                  (.to-tuple "pickle"))
      loader (WebLoader dataset :num-workers 0))

(print (tree-map #%(. %1 shape) (torch2jax (next (iter loader)))))

(setv torch2jax (tree-map #%(np.array (.numpy %1)))
      latents (map #%(-> %1 torch2jax ravel-pytree (get 0)) loader))
(print latents.shape)
        
;(setv pca (-> (PCA :n-components 100)
;            (.fit latents)))
(do
  (plt.semilogy (. pca explained_variance_ratio_))
  (plt.show))
(do
    (plt.imshow kernel)
    (plt.show))

