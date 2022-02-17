(import jax
        jax [numpy :as np]
        matplotlib [pyplot :as plt])

(defn sphere [x [radius 1.0]]
  (- (np.linalg.norm x) radius))

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

(let
  [ray-origin (np.array [0.0 0.0 -2.0])
   [u v] (np.meshgrid (np.linspace -1.0 1.0 1280)
                      (np.linspace -1.0 1.0 1280))
   uvw (np.dstack [u v (np.ones-like u)])
   ray-dir (/ uvw (np.linalg.norm uvw :axis -1 :keepdims True))
   ray-dir (.reshape ray-dir [-1 3])
   ray-origin (jax.lax.broadcast ray-origin [(get ray-dir.shape 0)])
   marcher (make-marcher sphere)
   _ (print (. ray-dir shape) (. ray-origin shape))
   vmarch (jax.vmap march :in-axes [0 0 None] :out-axes 0)
   res-pos (.reshape (vmarch ray-origin ray-dir marcher) [(get u.shape 0) (get u.shape 0) 3])
   res-vis (/ res-pos (np.linalg.norm res-pos :axis -1 :keepdims True))]
  (setv res res-vis))

(require hyrule *)

(import ngp)
(setv features (-> (jax.random.uniform ngp.KEY [ngp.T 2])))

(ngp.mlp-3 res (jax.random.uniform ngp.KEY [(* 2 3 64)]))

(plt.imshow res)
(plt.show)



