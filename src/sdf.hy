(import os [environ]
        functools [partial])
(setv (get environ "XLA_PYTHON_CLIENT_PREALLOCATE") "false")

(import jax
        jax [numpy :as np]
        numpy :as onp
        matplotlib :as mpl
        matplotlib [pyplot :as plt]
        cv2 [imshow waitKey destroyAllWindows]
        jax.example-libraries [stax])

(import ngp)

(require hyrule *)

(destroyAllWindows)

(defn sphere [x [radius 0.5]]
  (- (np.linalg.norm (- x 0.5) :axis -1 :keepdims True) radius))

(defn make-marcher [scene]
  (defn march-step [step arg-pack]
    (let [[prev-dist ray-origin ray-dir] arg-pack
          dist (scene ray-origin)]
      (, dist (+ ray-origin (* ray-dir dist)) ray-dir)))
  march-step)

(defn stop-marching [arg-pack]
  (| (< 1e-4 (get arg-pack 0))
     (> 1e3 (get arg-pack 0))))

(defn march [ray-origin ray-dir scene]
  (for [_ (range 0 30)]
    (setv ray-origin (+ ray-origin (* ray-dir (scene ray-origin)))))
  ray-origin)

(defn ϕ [d]
  (hy.pyops.reduce (fn [prev _] (** (+ 1 prev) (/ 1 (+ 1 d))))
          (range 0 100) 2.0))


(setv PLASTIC (lfor n-dim (range 0 7)
                  (-> (** (/ 1.0 (ϕ n-dim)) (np.arange 1 (+ 1 n-dim)))
                      (% 1.0) 
                      (np.expand-dims 0))))


(defn quasirandom [n n-dim [seed 0.5]]
  "R_d low discrepancy sequence from 
http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/"
  (-> (np.arange 1 (+ n 1))
      (np.expand-dims -1)
      (* (get PLASTIC n-dim))
      (+ seed)
      (% 1.0)))

(defn mse [x y]
  (.mean (** (- x y) 2)))

(defn iou [x y]
  (/ (* x y)))

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
  (print (. (np.arange cmap.N) shape))
  (setv my-cmap (cmap (np.arange cmap.N))
        (ncut my-cmap : -1) (** (np.linspace -1 1 cmap.N) 2))
 (mpl.colors.ListedColormap my-cmap))



(defn hash-feature-encoding [levels T]
  (stax.serial 
   (stax.FanOut (+ 1 levels))
   (stax.parallel 
     stax.Identity
     #* (lfor level (range 0 levels) 
           (ngp.HashEncodedFeatures T 2 level)))
   (stax.FanInConcat -1)))

(with-decorator jax.jit
  (defn train-step [weights x y]
    (setv get-grad (jax.value-and-grad (fn [weights x y]
                                        (mse y (mlp weights x)))))
    (get-grad weights x y)))
 
(do
 (defn Mlp []
    (stax.serial
      (hash-feature-encoding 8 (** 2 14))
      (stax.Dense 64) stax.Relu ; (stax.elementwise np.sin)
      (stax.Dense 64) stax.Relu
      (stax.Dense 64) stax.Relu
      (stax.Dense 1)))



 (setv [init-weights mlp] (Mlp)))


(do
  (setv 
        [out-shape weights] (init-weights ngp.KEY (, 3)) 
        optimizer (optax.adam 3e-4 :eps 1e-15)
        opt-state (.init optimizer weights)
        losses []
        vis (fn []
              (do
               (setv 
                     fig (plt.figure))
               (plot-density sphere (.add-subplot fig 1 2 1 :projection "3d"))
               (plot-density (partial mlp weights) (.add-subplot fig 1 2 2 :projection "3d")))))
               
  (for [epoch (range 0 100)]
    (setv seed (+ 0.5 (/ epoch 1000.0))
          xs (quasirandom (** 30 3) 3 :seed seed)
          ys (sphere xs)
          [loss grad] (train-step weights xs ys)
          [updates opt-state] (.update optimizer grad opt-state)
          weights (optax.apply-updates weights updates)) 
    (print loss)
    (.append losses (.item loss))
    (when (= 0 (% epoch 10))
      (do
        (vis)
        (show "training"))))
  
  (plt.plot losses)
  (show "loss-curve"))
  
  
  
  
      

     







(let [fig (plt.figure)]
  (plot-density sphere (.add-subplot fig 1 2 1 :projection "3d"))
  (plot-density (partial mlp weights) (.add-subplot fig 1 2 2 :projection "3d"))
  (plt.show))


(do  
  (setv
   ray-origin (np.array [[0.5 0.5 -1.0]])

   [u v] (np.meshgrid (np.linspace -1.0 1.0 32)
                      (np.linspace -1.0 1.0 32))
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
  
