(import 
  jax
  jax [numpy :as np]
  matplotlib [use :as mpl-use pyplot :as plt]
  functools [partial])
(require hyrule *)

(defn ? [f]
  (help f)) 

(import hash [ngp-hash R3-hash hash-without-sine])

(defn smoothstep [t]
  (* t t (- 3 (* 2 t))))

(setv KEY (jax.random.PRNGKey 2022))


(defn grid-resolution [level growth-factor coarsest]
  ((? np.floor)) 
  (* coarsest (** growth-factor level)))

(defn voxel-idxs [coord level]
  (let [lower (np.floor (* coord 16 (** 1.3819129 level)))
        lower (np.expand-dims lower 0)]
    (+ 0.5 lower (np.array [[0 0 0]
                            [1 0 0]
                            [0 1 0]
                            [0 0 1]
                            [1 1 0]
                            [0 1 1]
                            [1 0 1]
                            [1 1 1]] :dtype lower.dtype))))

(defn hash-encoded-features [coord features level [hasher ngp-hash]]
  (let [voxels (voxel-idxs coord level)
        T (len features)
        interpolate-by 
          (np.linalg.norm (- (np.expand-dims coord 0) voxels) 
                          :axis -1 :keepdims True)
        feature-idxs (hasher (.astype voxels np.uint32) T)
        voxel-feats (get features feature-idxs)]
   (np.sum (* interpolate-by voxel-feats) :axis 0)))


(defn HashEncodedFeatures [num-features feature-dim level [hasher ngp-hash] [weights-scale 1e-4]]
  (defn init-fn [rng input-shape]
    (setv output-shape (+ (cut input-shape 0 -1) (, feature-dim))
          params (jax.random.uniform rng [num-features feature-dim] 
                                     :minval (- weights-scale) :maxval weights-scale))
    (, output-shape (dict :hash params)))
  (defn apply-fn [params coord #** kwargs]
    ((jax.vmap hash-encoded-features [0 None None None]) 
     coord (:hash params) level hasher))
  [init-fn apply-fn])

