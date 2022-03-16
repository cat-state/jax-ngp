(import os [environ])
(setv (get environ "XLA_PYTHON_CLIENT_PREALLOCATE") "false")

(import jax
        jax [numpy :as np random]
        numpy :as onp
        matplotlib :as mpl
        matplotlib [pyplot :as plt]
        cv2 [imshow waitKey destroyAllWindows]
        jax.example-libraries [stax]
        jax.tree-util [Partial :as partial])

(import ngp
        hash [quasirandom])

(require hyrule *)



(defn sphere [x [radius 0.5]]
  (- (np.linalg.norm (- x 0.5) :axis -1 :keepdims True) radius))


(defn blobby [x]
  (np.minimum (sphere x) (* 0.0 (+ 1.0 (np.sin (* 5.0 x))))))

(defn make-marcher [scene]
  (defn march-step [step arg-pack]
    (let [[prev-dist ray-origin ray-dir] arg-pack
          dist (scene ray-origin)]
      (, dist (+ ray-origin (* ray-dir dist)) ray-dir)))
  march-step)

(defn stop-marching [arg-pack]
  (| (< 1e-4 (get arg-pack 0))
     (> 1e3 (get arg-pack 0))))

#@(jax.jit 
    (defn march [ray-origin ray-dir scene]
      (for [_ (range 0 30)]
        (setv ray-origin (+ ray-origin (* ray-dir (scene ray-origin))))) 
      ray-origin))



(defn mse [x y]
  (.mean (** (- x y) 2)))

(defn iou [x y]
  (/ (* x y) x))

(defn smape [x y]
  (setv diff (- y x)
        factor (/ 1.0 (* 0.5 (+ (+ (abs x) (abs y)) 1e-2))))
  (.mean (* factor (abs diff))))

(defn mape [target pred]
  (.mean (/ (abs (- pred target)) 
            (+ (abs target) 1e-2))))


(import optax)

(defn to-01 []
  (defn init-fn [rng input-shape]
    [input-shape (,)])
  (defn apply-fn [_ x #** kwargs]
    (-> x (* 0.5) (+ 0.5)))
  [init-fn apply-fn])


(defn show [window-name]
  (. (plt.gcf) canvas draw)
  (setv fig (plt.gcf)
        canvas (. fig canvas))
  (.draw canvas)

  (imshow
    window-name
    (-> (onp.frombuffer (.tostring-rgb canvas) :dtype onp.uint8)

      (.reshape (+ (ncut (.get-width-height canvas) ::-1) (, 3)))
      (ncut : : ::-1)
      (onp.ascontiguousarray)))
  (plt.clf)
  (plt.close)
  (waitKey 1))




  

(defn plot-density [xyz->c [ax None]]
  (setv
     my-cmap (make-alpha plt.cm.viridis)
     ax (if (is None ax)
          (.add-subplot (plt.figure) :projection "3d")
          ax)
     xyz (quasirandom (** 30 3) 3)
     dist (xyz->c xyz)
     [x y z] (np.split xyz 3 :axis -1))
  (.scatter ax (.ravel x) (.ravel y) (.ravel z) :c (.ravel dist) :cmap my-cmap))


(import mpl-toolkits.mplot3d [Axes3D])

; from https://jax.readthedocs.io/en/latest/notebooks/convolutions.html

(defn make-alpha [cmap]
  (setv my-cmap (cmap (np.arange cmap.N))
        (ncut my-cmap : -1) (** (np.linspace -1 1 cmap.N) 2))
 (mpl.colors.ListedColormap my-cmap))


(defn hash-feature-encoding [levels T]
  (stax.serial 
   (stax.FanOut (+ 1 levels))
   (stax.parallel 
     stax.Identity #* (lfor level (range 0 levels) 
                       (ngp.HashEncodedFeatures T 2 level :hasher ngp.R3-hash)))
   (stax.FanInConcat -1)))

(with-decorator jax.jit
  (defn train-step [weights x y]
    (setv get-grad (jax.value-and-grad (fn [weights x y]
                                        (mape y (mlp weights x)))))
    (get-grad weights x y)))


(defn residual [#* inner]
  (stax.serial
    (stax.FanOut 2)
    (stax.parallel
      stax.Identity (stax.serial #* inner)
     stax.FanInSum)))

(defn siren [x]
  (np.sin (* 30.0 x)))

(defn Siren []
  (defn init-fn [rng input-shape] [input-shape (,)])
  (defn apply-fn [_ x #** kwargs] (siren x))
  [init-fn apply-fn])

(setv ortho-init (jax.nn.initializers.orthogonal))
(defn siren-init [rng shape]
  (setv [in-dim out-dim] shape)
  (random.uniform rng shape :minval (- (-> in-dim (/ 6.0) (** 0.5)))
                            :maxval (-> in-dim (/ 6.0) (** 0.5)))) 


(defn id-mlps [out-dim]
  (defn init-fn [rng input-shape]
    (setv [#* hd in-dim] input-shape
          [w1 w2] (random.split rng 2))
    (, #* hd (, out-dim)
     [(ortho-init w1 [in-dim out-dim])
      (ortho-init w2 [out-dim out-dim])]))
  (defn apply-fn [ws x #** kwargs]
    (setv [w1 w2] ws)
    (-> x (@ w1) jax.nn.relu (@ w2)))
  [init-fn apply-fn])

(defn suml []
  (defn init-fn [rng input-shape]
    (, (cut input-shape 0 -1) (,)))
  (defn apply-fn [_ x #** kwargs]
    (.sum x :axis -1 :keepdims True))
  [init-fn apply-fn])

(do
 (defn Mlp []
    (stax.serial
      (hash-feature-encoding 16 (** 2 19))
      (id-mlps 64) (suml)))

 (setv [init-weights mlp] (Mlp))) 
(import tqdm [tqdm])

(import jax.tree-util [tree-map])

(require hash [vectorized])

(vectorized defn rot-xy [vec angle] "(3),()->(3)"
  (@ vec (np.array [[(np.cos angle) (- (np.sin angle)) 0.0]
                    [(np.sin angle) (np.cos angle) 0.0]
                    [0.0 0.0 1.0]]))) 

(vectorized defn rot-yz [vec angle] "(3),()->(3)"
  (@ vec (np.array [[1.0 0.0 0.0]
                    [0.0 (np.cos angle) (- (np.sin angle))]
                    [0.0 (np.sin angle) (np.cos angle)]]))) 


(defn normalize [x]
  (/ x (np.linalg.norm x :axis -1 :keepdims True)))

#@(jax.jit 
    (defn normals [x scene]
      "Compute SDF normals by central differences using tetrahedron offsets
      from https://www.iquilezles.org/www/articles/normalsSDF/normalsSDF.htm"
      (setv tet (np.array [[ 1 -1 -1]
                           [-1 -1  1]
                           [-1  1 -1]
                           [ 1  1  1]]))
      (normalize
        (+ #* (lfor offset tet (* offset (scene (+ x offset))))))))
      
  
 
(defn sample-ray-points [key ray-origin ray-dir n]
  (-> (* (** 3 0.5) (jax.random.uniform key [n #* ray-dir.shape]) (ncut ray-dir None))
     (+ (.reshape ray-origin [1 1 3]))
     (% 1.0)))


(defn march-vis [scene march-scene rng]
  (setv
     ray-origin (np.array [[0.5 0.5 -1.0]])
     [u v] (np.meshgrid (np.linspace -1.0 1.0 320)
                        (np.linspace -1.0 1.0 320))
     uvw (np.dstack [u v (np.ones-like u)])
     ray-dir (/ uvw (np.linalg.norm uvw :axis -1 :keepdims True))

     ray-origin (rot-xy ray-origin (random.uniform rng []))
     ray-dir (rot-xy ray-dir (random.uniform rng []))
     ray-dir (.reshape ray-dir [-1 3])
     ; res-flat (march ray-origin ray-dir scene)
     res-flat (march-scene ray-origin ray-dir)
     res-normal (normals res-flat scene)
     ; [_ res-normal] (jax.jvp (partial mlp weights) [res-flat] [(np.ones-like res-flat)])
     unflat (fn [x] (.reshape x [(get u.shape 0) (get u.shape 1) -1]))

     res-vis (np.where (< (scene res-flat) 0.01)
                 res-flat
                (np.zeros-like res-normal))
     res-vis (unflat res-vis)) 
  (imshow "march vis" (onp.asarray res-vis))
  (waitKey 1))

(defmacro "#ignore" [form])

(defn cat [x y]
  (tuple (+ (list x) (list y))))

(do

  (setv 
        [out-shape weights] (init-weights ngp.KEY (, 3))
        _ (print (tree-map (fn [t] (not-in "hash" t)) weights
                                     :is-leaf (fn [n] (in "hash" n)))) 
        adamw   (optax.adamw 
                     ; 1e-4
                     (optax.exponential-decay :init-value 1e-4 :transition-steps 250 :decay-rate 0.1)
                     :b1 0.9 :b2 0.99 :eps 1e-15
                     :weight-decay 1e-6
                     :mask (tree-map (fn [t] (not-in "hash" t)) weights
                                     :is-leaf (fn [n] (in "hash" n))))

        optimizer (optax.chain adamw)  ; (optax.chain (optax.lamb 1e-4 :b2 0.99 :eps 1e-15))
        opt-state (.init optimizer weights)
        losses []
        march-mlp (jax.jit (fn [ro rd weights]
                            (march ro rd (partial mlp weights))))
        
        ray-origin (np.array [[0.5 0.5 -2.0]])
        [u v] (np.meshgrid (np.linspace -1.0 1.0 100)
                          (np.linspace -1.0 1.0 100))
        uvw (np.dstack [u v (np.ones-like u)])
        ray-dir (/ uvw (np.linalg.norm uvw :axis -1 :keepdims True))
        ray-dir (.reshape ray-dir [-1 3])      
        vis (fn []
              (do
               (setv fig (plt.figure))
               (plot-density sphere (.add-subplot fig 1 2 1 :projection "3d"))
               (plot-density (partial mlp weights) (.add-subplot fig 1 2 2 :projection "3d")))))
               
  (for [epoch (tqdm (range 500))]
    (setv seed (+ 0.5 (/ epoch 20000.0))
          [key rd longitude latitude] (jax.random.split (jax.random.PRNGKey epoch) 4)
          ; xs (quasirandom 100000 3 :seed seed)
          ; xs (jax.random.uniform (jax.random.PRNGKey epoch) [100000 3])
          uvw (np.concatenate [(jax.random.uniform rd [2048 1 2] 
                                                   :minval -1 :maxval 1)
                               (np.ones [2048 1 1])] :axis -1)
          ray-dir (/ uvw (np.linalg.norm uvw :axis -1 :keepdims True))
          ray-dir (.reshape ray-dir [-1 3])
          ray-origin (rot-xy ray-origin (jax.random.uniform longitude []))
          ray-dir (rot-xy ray-dir (jax.random.uniform longitude []))
          xs (sample-ray-points key ray-origin ray-dir 1024)
          xs (.reshape xs [-1 3])
          ; xs (np.concatenate [xs (jax.random.uniform key [10000 3])] :axis 0)
          ; ys (np.tanh (sphere xs))
          ys (sphere xs)
          [loss grad] (train-step weights xs ys)
          [updates opt-state] (.update optimizer grad opt-state :params weights)
          new-weights (optax.apply-updates weights updates)
          weights (optax.incremental-update new-weights weights 0.05))
    (unless (% epoch 50) 
            (march-vis (partial mlp weights) (partial march-mlp :weights weights) longitude)) 


    (.append losses (.item loss))
    (when (and True (= 0 (% epoch 500)))
      (do)))
  (print weights)
  (plt.semilogy losses)
  (show "losses")



  #ignore (let [fig (plt.figure)]
           (plot-density sphere (.add-subplot fig 1 2 1 :projection "3d"))
           (plot-density (partial mlp weights) (.add-subplot fig 1 2 2 :projection "3d"))
           (plt.show))


  (import time [time])
  (do
    (setv t0 (time))
    (setv jmarch (jax.jit (partial march :scene (partial sphere))))
    (print (- (time) t0))
    (jax.block-until-ready 
      (jmarch ray-origin ray-dir))
    (print (- (time) t0)))

  (do  
    (setv
     ray-origin (np.array [[0.5 0.5 -1.0]])

     [u v] (np.meshgrid (np.linspace -1.0 1.0 320)
                        (np.linspace -1.0 1.0 320))
     uvw (np.dstack [u v (np.ones-like u)])
     ray-dir (/ uvw (np.linalg.norm uvw :axis -1 :keepdims True))
     ray-dir (.reshape ray-dir [-1 3])
     res-flat (march-mlp ray-origin ray-dir weights) 
     res-pos (.reshape res-flat
                       [(get u.shape 0) (get u.shape 0) 3])
     res-vis (/ res-pos (np.linalg.norm res-pos :axis -1 :keepdims True)))
    (plt.imshow res-pos)
    (plt.show)))

(destroyAllWindows)
