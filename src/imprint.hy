(import jax
        jax [numpy :as np
             random]
        jax.example-libraries [stax]
        jax.tree-util [tree-map
                       Partial :as partial]
        numpy :as onp
        matplotlib [pyplot :as plt]
        seaborn
        pandas
        optax)

(import ngp [voxel-idxs HashEncodedFeatures])
(import hash [ngp-hash R3-hash inv-R3-hash hash-without-sine])
(import sdf [show sphere])

(require hyrule *)

(defn zipmap [f #* its]
  (map (fn [a] (f #* a)) (zip #* its)))

(setv KEY (random.PRNGKey 2022))

(defn i** [b x] (int (** b x)))

(defn raster [x voxels] 
  (np.ravel-multi-index (.split (.astype x np.uint32) 3 :axis -1) [voxels voxels voxels] 
                        :mode "wrap"))


(defn ortho-dense [out-dim]
  (defn init-fn [rng input-shape]
    (, (+ (cut input-shape 0 -1) (, out-dim))
       ((jax.nn.initializers.orthogonal) rng [(get input-shape -1) out-dim])))
  (defn apply-fn [params x #** kwargs]
    (@ x params))
  [init-fn apply-fn])

(setv m ((jax.nn.initializers.orthogonal) KEY [2 2])
      m2 ((jax.nn.initializers.orthogonal) (random.fold-in KEY 3) [4 4])
      [init-hef apply-hef] (stax.serial 
                            (stax.FanOut 2) 
                            (stax.parallel #* (lfor level (range 2) (HashEncodedFeatures 1000 2 level :weights-scale 1e-4)))
                            (stax.FanInConcat -1)))


(setv x (np.array [[1.0 0.0 0.0]])
      y (np.array [[0.0 0.0 0.0 1.0]])
      [_ hef] (init-hef KEY (, 3)) 
      
      ; sgd (optax.chain (optax.sgd 1.0) (optax.masked (optax.set-to-zero) (tree-map (fn [n] (not-in "hash" n)) hef :is-leaf #%(= (type %1) dict))))
      sgd (optax.sgd 1)
      sgd-state (.init sgd hef)
      _ (apply-hef hef x)
      fwd (fn [hef x y]          
            (.mean (np.square (- (@ (apply-hef hef x) m2) y))))
      args [hef x y]
      ; g ((jax.linear-transpose #%(@ (apply-hef %1 %2) m2) hef x) y)
      [v g] ((jax.value-and-grad fwd) #* args)
      ; [updates sgd-state] (.update sgd (tree-map (fn [t] (/ 1.0 t)) g) sgd-state hef)
      [updates sgd-state] (.update sgd g sgd-state hef)
      ;  (get updates -1) (* 0 (get updates -1))
      hef2 (optax.apply-updates hef updates)
      ff (@ (apply-hef hef2 x) m2)
      _ (print ff))





(defn raster-hash [x T]
  (-> (raster x (int (np.ceil (** T (/ 1 3))))) 
      (% T)
      (.squeeze))) 


(defn imprint [weights features idxs y]
  (.set (get features.at idxs)
        (@ y (np.linalg.pinv weights))))

(defn mr-imprint [weights levels idxs y]
  (setv x (@ y (np.linalg.pinv weights))
        ;_ (print weights.shape y.shape)
        ;[x _ _ _] (np.linalg.lstsq weights.T y.T)
        ;_ (print x.shape)
        ;x x.T
        x (np.split x (len levels) :axis -1)
        x (zipmap average-colliding levels idxs x))
  (lfor [level xs idx] (zip levels x idxs)
      (.set (get level.at idx) xs)))

(defn average-colliding [feats idxs y]
  (setv z (np.zeros-like feats :dtype np.float32)
        ym (np.zeros-like feats :dtype np.float32)
        z (.add (get z.at idxs) 1.0)
        ym (.add (get ym.at idxs) (np.log1p y)))
  (np.expm1 (/ (get ym idxs) (get z idxs))))

(defn multilevel-hash [levels hasher x T]
  (-> x  (ncut ... None)
     (* (-> (np.geomspace 16 2048 :num 16) (np.ceil) (ncut None)))
     (np.floor)
     (.transpose 2 0 1)
     (hasher T)))

(np.expm1 (np.log1p 50))

(do
  (setv 
    y (np.array [[0 1] [1 2] [50 50]])
    idxs (np.array [0 0 2]))
  (print y)
  (print #* (zipmap average-colliding [idxs] [y] [y])))

(defn gt-if-close [x gt]
  (np.where (np.isclose x gt) gt x))

(defn subplots [h w #** kwargs]
  (get (plt.subplots h w #** kwargs) 1))


(defmain []
  (setv results 
   (lfor hasher [raster-hash R3-hash inv-R3-hash hash-without-sine ngp-hash]
       nx (np.linspace 3 5 :num 5)
       scene ["sphere" "random"]
       levels [1 8 16]
     (do  
       (setv
         T (** 2 12)
         [w-key x-key h-key y-key] (random.split KEY 4)
         weights (random.uniform w-key [64 1])
         feats (lfor key (random.split h-key levels) (random.uniform key [T (// 64 levels)]))
         x (random.uniform x-key [(i** 10 nx) 3])
         idxs (multilevel-hash levels hasher x T)
         y-true (if (= scene "sphere") (sphere x) (random.uniform y-key [(i** 10 nx) 1]))
         imprinted (mr-imprint weights feats idxs y-true)
         f2 (np.concatenate (lfor [idx level] (zip idxs imprinted) (get level idx)) :axis -1)
         y-true (ncut y-true : 0)
         pred (ncut (@ f2 weights) : 0)
         err (abs (/ (abs (- pred y-true)) (+ (abs pred) 1e-2)))
         ; err (abs (/ (abs (- pred y-true)) (+ (abs (/ (+ y-true pred) 2)) 1e-2)))
         load-factor (/ (. (np.unique (raster (.astype (* x 2048) np.int32) 2048)) size)
                        T)
         geo-mape (-> err np.log1p np.mean np.expm1 (np.clip 0 2))
         mape (.mean err))
      (print (. hasher __name__) (i** 10 nx) load-factor mape geo-mape)
      {"log load factor" (.item (np.log2 load-factor))
       "load factor" load-factor 
       "scene" scene "levels" levels
       "nx" (str (.item nx))
       "err" (onp.array err)
       "mape" (.item geo-mape) "hasher" (. hasher __name__)})))


  (setv results (pandas.DataFrame.from-records results))
  (print results)
  (-> (seaborn.FacetGrid results :row "levels" :col "scene" :hue "hasher")
     (.map seaborn.lineplot "log load factor"  "mape")
     (.add-legend)) 

  (plt.show)) 
