(import os [environ])
(setv (get environ "XLA_PYTHON_CLIENT_PREALLOCATE") "false")

(import jax
        jax [numpy :as np]
        matplotlib :as mpl
        matplotlib [pyplot :as plt]
        jax.example-libraries [stax])

(import ngp)

(require hyrule *)


(defn sphere [x [radius 0.5]]
  (- (np.linalg.norm x :axis -1) radius))

(defn make-marcher [scene]
  (defn march-step [step arg-pack]
    (let [[prev-dist ray-origin ray-dir] arg-pack
          dist (scene ray-origin)]
      (, dist (+ ray-origin (* ray-dir dist)) ray-dir)))
  march-step)

(defn stop-marching [arg-pack]
  (| (< 1e-4 (get arg-pack 0))
     (> 1e3 (get arg-pack 0))))

(defn march [ray-origin ray-dir marcher]
  (let [init-val (, 1000.0 ray-origin ray-dir)
        [dist ray-end _] (jax.lax.fori-loop
                           0 30 marcher init-val)]
    ray-end))

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

(import optax)

(defn Mlp []
  (stax.serial
    (stax.Dense 64) stax.Relu
    (stax.Dense 64) stax.Relu
    (stax.Dense 64) stax.Relu
    (stax.Dense 1)))


(setv [init-weights mlp] (Mlp))

(with-decorator jax.jit
  (defn train-step [weights x y]
    (setv get-grad (jax.value-and-grad (fn [weights x y]
                                        (mse y (mlp weights x)))))
    (get-grad weights x y)))
 
(do
  (setv 
        [out-shape weights] (init-weights ngp.KEY (, 3)) 
        optimizer (optax.adam 3e-3)
        opt-state (.init optimizer weights)
        losses [])
  (for [epoch (range 0 100)]
    (setv xs (-> (quasirandom 10000 3) (* 2) (- 1))
          ys (sphere xs 0.5)
          [loss grad] (train-step weights xs ys)
          [updates opt-state] (.update optimizer grad opt-state)
          weights (optax.apply-updates weights updates)) 
    (print loss)
    (.append losses (.item loss)))
  (plt.plot losses)
  (plt.show))

(.std weights)
(import mpl-toolkits.mplot3d [Axes3D])

; from https://jax.readthedocs.io/en/latest/notebooks/convolutions.html

(defn make-alpha [cmap]
  (print (. (np.arange cmap.N) shape))
  (setv my-cmap (cmap (np.arange cmap.N))
        (ncut my-cmap : -1) (** (np.linspace -1 1 cmap.N) 2))
 (mpl.colors.ListedColormap my-cmap))


(defn plot-density [xyz->c [ax None]]
  (setv
     my-cmap (make-alpha plt.cm.viridis)
     ax (if (is None ax)
          (.add-subplot (plt.figure) :projection "3d")
          ax)
     xyz (-> (quasirandom (** 30 3) 3) (* 2) (- 1))
     dist (xyz->c xyz)
     [x y z] (np.split xyz 3 :axis -1))
  (.scatter ax (.ravel x) (.ravel y) (.ravel z) :c (.ravel dist) :cmap my-cmap))

(let [fig (plt.figure)]
  (help (. fig add-subplot))
  (plot-density (partial sphere :radius 0.5) (.add-subplot fig 2 1 1 :projection "3d"))
  (plot-density (partial mlp weights) (.add-subplot fig 2 1 2 :projection "3d"))
  (plt.show))

(import functools [partial])

(setv mlp-sdf (partial ngp.mlp-sdf 
                :weights (jax.random.uniform ngp.KEY [(+ (* 1 64) 
                                                         (* 3 64))])))

(let
  [ray-origin (np.array [0.0 0.0 -1.0])
   [u v] (np.meshgrid (np.linspace -1.0 1.0 128)
                      (np.linspace -1.0 1.0 128))
   uvw (np.dstack [u v (np.ones-like u)])
   ray-dir (/ uvw (np.linalg.norm uvw :axis -1 :keepdims True))
   ray-dir (.reshape ray-dir [-1 3])
   ray-origin (jax.lax.broadcast ray-origin [(get ray-dir.shape 0)])
   vmarch (jax.vmap march :in-axes [0 0 None] :out-axes 0)
   res-pos (.reshape (vmarch ray-origin ray-dir (partial mlp weights)) 
                     [(get u.shape 0) (get u.shape 0) 3])
  ; res-pos (.reshape (vmarch ray-origin ray-dir (partial sphere :radius 0.5)) 
  ;                   [(get u.shape 0) (get u.shape 0) 3])
   res-vis (/ res-pos (np.linalg.norm res-pos :axis -1 :keepdims True))]
  (plt.imshow res-pos)
  (plt.show))

(setv features (jax.random.uniform ngp.KEY [ngp.T 2] :minval -1e-4 :maxval 1e-4))

(->
  (ngp.mlp-sdf res (jax.random.uniform ngp.KEY [(+ (* 1 64) 
                                                   (* 3 64))]))
  (plt.imshow)
  (print))
(plt.show)
(plt.imshow res)
(plt.show)



