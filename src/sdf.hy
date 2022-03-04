(import os [environ])
(setv (get environ "XLA_PYTHON_CLIENT_PREALLOCATE") "false")

(import jax
        jax [numpy :as np]
        numpy :as onp
        matplotlib :as mpl
        matplotlib [pyplot :as plt]
        cv2 [imshow waitKey destroyAllWindows]
        jax.example-libraries [stax]
        jax.tree-util [Partial :as partial])

(import ngp
        hash [quasirandom])

(require hyrule *)

(destroyAllWindows)

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
   (stax.FanOut levels)
   (stax.parallel 
     #* (lfor level (range 0 levels) 
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

(do
 (defn Mlp []
    (stax.serial
      (hash-feature-encoding 16 (** 2 19))
      (stax.Dense 64) stax.Relu
      (stax.Dense 64) stax.Relu
      (stax.Dense 1))))


(setv [init-weights mlp] (Mlp))
(import tqdm [tqdm])

(import jax.tree-util [tree-map])

(require hash [vectorized])

(vectorized defn rot-xy [vec angle] "(2),()->(2)"
  (@ vec (np.array [[(np.cos angle) (- (np.sin angle)) 0.0]
                    [(np.sin angle) (np.cos angle) 0.0]
                    [0.0 0.0 1.0]]))) 

(vectorized defn rot-yz [vec angle] "(2),()->(2)"
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
      
  
  

(defn march-vis [scene]
  (setv
     ray-origin (np.array [[0.5 0.5 -1.0]])
     [u v] (np.meshgrid (np.linspace -1.0 1.0 320)
                        (np.linspace -1.0 1.0 320))
     uvw (np.dstack [u v (np.ones-like u)])
     ray-dir (/ uvw (np.linalg.norm uvw :axis -1 :keepdims True))
     ray-dir (.reshape ray-dir [-1 3])
     marcher  (fn [ro rd] (march ro rd (partial mlp weights)))
     res-flat (march ray-origin ray-dir scene)
     res-normal (normals res-flat scene)
     ; [_ res-normal] (jax.jvp (partial mlp weights) [res-flat] [(np.ones-like res-flat)])
     unflat (fn [x] (.reshape x [(get u.shape 0) (get u.shape 1) -1]))

     res-vis (np.where (< (scene res-flat) 0.01)
                 res-flat
                (np.zeros-like res-normal))
     res-vis (unflat res-vis)) 
  res-vis)

(import time [time])
(jax.profiler.start-trace "/tmp/tensorboard")
(setv t0 (time)
      tst (jax.jit march-vis)
      scene (partial sphere))
(setv _ (tst scene))
(for [x (range 30)]
  (setv ret (tst scene)))
(jax.block-until-ready ret)
(print (- (time) t0))
(jax.profiler.stop-trace)
(raise SystemExit)
 ; (imshow "march-vis" (onp.asarray res-vis))
 ; (waitKey 1))



(do
  (setv 
        [out-shape weights] (init-weights ngp.KEY (, 3))
        _ (print (tree-map (fn [t] (not-in "hash" t)) weights
                                     :is-leaf (fn [n] (in "hash" n)))) 
        adamw   (optax.adamw 
                     ; 1e-4
                     (optax.exponential-decay :init-value 1e-4 :transition-steps 1000 :decay-rate 0.1)
                     :b1 0.9 :b2 0.99 :eps 1e-15
                     :weight-decay 1e-6
                     :mask (tree-map (fn [t] (not-in "hash" t)) weights
                                     :is-leaf (fn [n] (in "hash" n))))

        optimizer adamw ; (optax.chain (optax.lamb 1e-4 :b2 0.99 :eps 1e-15))


        opt-state (.init optimizer weights)
        losses []
        
        ray-origin (np.array [[0.5 0.5 -1.0]])

        [u v] (np.meshgrid (np.linspace -1.0 1.0 512)
                          (np.linspace -1.0 1.0 512))
        uvw (np.dstack [u v (np.ones-like u)])
        ray-dir (/ uvw (np.linalg.norm uvw :axis -1 :keepdims True))
        ray-dir (.reshape ray-dir [-1 3])      
        vis (fn []
              (do
               (setv fig (plt.figure))
               (plot-density sphere (.add-subplot fig 1 2 1 :projection "3d"))
               (plot-density (partial mlp weights) (.add-subplot fig 1 2 2 :projection "3d")))))
               
  (for [epoch (tqdm (range 0 11000))]
    (setv seed (+ 0.5 (/ epoch 20000.0))
          ; xs (quasirandom 100000 3 :seed seed)
          xs (if (and (> epoch 10) (= 0 (% epoch 2)))
               (do
                 (setv angle (jax.random.uniform (jax.random.PRNGKey epoch) [1]))
                 (march (rot-xy ray-origin angle) (rot-xy ray-dir angle) (partial mlp weights)))
               (jax.random.uniform (jax.random.PRNGKey epoch) [300000 3]))  
          ys (np.tanh (sphere xs))
          [loss grad] (train-step weights xs ys)
          [updates opt-state] (.update optimizer grad opt-state :params weights)
          weights (optax.apply-updates weights updates))
    ; (march-vis (partial mlp weights)) 


    (.append losses (.item loss))
    (when (and True (= 0 (% epoch 500)))
      (do
        (march-vis (partial mlp weights)))))
  (plt.semilogy losses)
  (plt.show))

(destroyAllWindows)

(let [fig (plt.figure)]
  (plot-density sphere (.add-subplot fig 1 2 1 :projection "3d"))
  (plot-density (partial mlp weights) (.add-subplot fig 1 2 2 :projection "3d"))
  (plt.show))


(import time [time])
(do
  (setv t0 (time))
  (jax.jit march)
  (print (- (time) t0))
  (jax.block-until-ready 
    ((jax.jit march) ray-origin ray-dir sphere))
  (print (- (time) t0)))

(do  
  (setv
   ray-origin (np.array [[0.5 0.5 -1.0]])

   [u v] (np.meshgrid (np.linspace -1.0 1.0 320)
                      (np.linspace -1.0 1.0 320))
   uvw (np.dstack [u v (np.ones-like u)])
   ray-dir (/ uvw (np.linalg.norm uvw :axis -1 :keepdims True))
   ray-dir (.reshape ray-dir [-1 3])
   marcher (jax.jit (fn [ro rd] (march ro rd (partial mlp weights))))
   res-flat (marcher ray-origin ray-dir) 
   res-pos (.reshape res-flat
                     [(get u.shape 0) (get u.shape 0) 3])
   res-vis (/ res-pos (np.linalg.norm res-pos :axis -1 :keepdims True)))
  (plt.imshow res-pos)
  (plt.show))

