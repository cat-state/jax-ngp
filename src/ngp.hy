(import 
  jax
  jax [numpy :as np]
  matplotlib [use :as mpl-use pyplot :as plt]
  functools [partial])
(require hyrule *
         util ["#dbg"])
(defn ? [f]
  (help f)) 

(import hash [ngp-hash R3-hash hash-without-sine])

(defn smoothstep [t]
  (* t t (- 3 (* 2 t))))

(setv KEY (jax.random.PRNGKey 2022))

(setv voxel-offsets 
  [(np.array [[]])
   (np.array [[0] [1]])
   (np.array [[0 0] 
              [1 0] 
              [0 1] 
              [1 1]] :dtype np.float32)
   (np.array [[0 0 0]
              [1 0 0]
              [0 1 0]
              [0 0 1]
              [1 1 0]
              [0 1 1]
              [1 0 1]
              [1 1 1]] :dtype np.float32)]
  voxel-scale (lfor v voxel-offsets (- 1 v)))
   
(import jax.scipy.ndimage [map-coordinates]) 

(defn interpolate [x grid]
  (-> ((jax.vmap map-coordinates [-1 None None None]) grid x.T 1 "wrap")
      (.transpose)))


(defn grid-resolution [level growth-factor coarsest]
  (* coarsest (** growth-factor level)))

(defn voxel-idxs [coord level]
  (let [scale (- (* 16 (** 1.4 level)) 1.0)
        pos (+ 0. (* coord scale))
        lower (np.floor pos)
        grid (+ lower (get voxel-offsets (get lower.shape -1)))
        fract (- pos grid)]
    [fract grid]))

#@((partial jax.jit :static-argnames "hasher")
   (defn hash-encoded-features [coord features level [hasher ngp-hash]]
    (let [[fract voxels] (voxel-idxs coord level)
          T (len features)
          interpolate-by (np.prod (- 1.0 (abs fract)) ;(smoothstep (- 1.0 (abs fract)))
                               :initial 1.0
                               :axis -1 :keepdims True)
          feature-idxs (hasher voxels T)
          voxel-feats (get features feature-idxs)]
      (np.sum (* interpolate-by voxel-feats) :axis 0))))


(defn HashEncodedFeatures [num-features feature-dim level [hasher ngp-hash] [weights-scale 1e-4]]
  (defn init-fn [rng input-shape]
    (setv output-shape (+ (cut input-shape 0 -1) (, feature-dim))
          params (jax.random.uniform rng [num-features feature-dim] 
                                     :minval (- weights-scale) :maxval weights-scale))
    (, output-shape (dict :hash params)))
  (defn apply-fn [params coord #** kwargs]
    ((jax.vmap hash-encoded-features [0 None None None]) 
     coord (:hash params) level hasher))
  [init-fn (jax.profiler.annotate-function apply-fn :name "hashy")])


(defn GridEncodedFeatures [side feature-dim]
  (defn init-fn [rng input-shape]
    (setv output-shape (+ (cut input-shape 0 -1) (, feature-dim))
          params (jax.random.uniform rng [(int side) (int side) 
                                          feature-dim]
                                     :minval -1e-4 :maxval 1e-4))
    (, output-shape (dict :hash params)))
  (defn apply-fn [params coord #** kwargs]
    (interpolate (* side coord) (:hash params)))
  [init-fn apply-fn])


(defn encoding [num-features feature-dim level [hasher ngp-hash]]
  (setv side (int (** num-features 0.5))
        hash-res (grid-resolution level 1.4 16))
  (if (< hash-res side)
    (GridEncodedFeatures hash-res feature-dim)
    (HashEncodedFeatures num-features feature-dim level hasher)))

