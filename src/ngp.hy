(import 
  jax
  jax [numpy :as np]
  matplotlib [use :as mpl-use pyplot :as plt])
(require hyrule *)
(defn show []
  (if (not (= __name__ "__console__"))
      (plt.show)))

(setv 
  T (** 2 14) 
  π_2 (% 2654435761 (** 2 31))
  π_3 805459861)

(defn spatial-hash [coord]
  (% (^ (get coord 0) 
        (^ (* (get coord 1) π_2) 
           (* (get coord 2) π_3))) 
    T))

(defn smoothstep [t]
  (* t t (- 3 (* 2 t))))

(defn grid-resolution [level growth-factor coarsest]
  (np.floor 
    (* coarsest (** growth-factor level))))

(defn voxel-idxs [coord level]
  (let [lower (np.floor (* coord (grid-resolution level 2.0 0.5)))
        lower (np.expand-dims lower 0)]
    (+ lower (np.array [[0 0 0]
                        [1 0 0]
                        [0 1 0]
                        [0 0 1]
                        [1 1 0]
                        [0 1 1]
                        [1 0 1]
                        [1 1 1]] :dtype lower.dtype))))

(defn hash-encoded-features [coord features level]
  (let [voxels (voxel-idxs coord level)
        interpolate-by 
          (np.linalg.norm (- (np.expand-dims coord 0) voxels) 
                          :axis -1 :keepdims True)
        feature-idxs (vhash (.astype voxels np.int32))
        voxel-feats (get features feature-idxs)]
   (np.sum (* interpolate-by voxel-feats) :axis 0)))

(defn mlp [x weights in-dim out-dim]
  (-> x
      (@ (.reshape (cut weights 0 (* in-dim 64)) [in-dim 64]))
      (jax.nn.relu)
      (@ (.reshape (cut weights (* 64 in-dim) None) [64 out-dim]))))

(defn mlp-sdf [x weights]
  (mlp x weights 3 1))

(defn mlp-3 [x weights]
  (mlp x weights 3 3))


(setv vhash (jax.vmap spatial-hash))
(setv v-hef (jax.vmap hash-encoded-features [0 None None]))
(setv vv-idxs (jax.vmap voxel-idxs [0 None]))

(setv xla-mlp (.xla-computation jax mlp-3))
(setv jit-mlp (.jit jax mlp-3))

(setv KEY (jax.random.PRNGKey 2022))

(if (= __name__ "__main__")
  (do
    (mpl-use "module://matplotlib-backend-kitty")
    (-> (xla-mlp (jax.random.uniform KEY [3])
                 (jax.random.uniform KEY [(* 3 64 2)]))
        (.as-hlo-text))
        ;(print)
        
    (-> (jit-mlp (jax.random.uniform KEY [3])
                 (jax.random.uniform KEY [(* 3 64 2)]))
        (print)) ; [25.845406 26.278118 26.071196]

    (setv vhash (jax.vmap spatial-hash)
          coords (-> (np.linspace 0 10000 1000 :dtype np.int32) 
                    (np.tile [3 1])
                    (.transpose 1 0)))
    (setv tst (get coords 59)
          vid (voxel-idxs tst 1)
          feats (-> (np.linspace 0 1 T) (np.tile [2 1]) (.transpose 1 0))
          fts (hash-encoded-features tst feats 1)
          _ (print (. vid shape) (. tst shape) (. fts shape)))

    (. (vv-idxs coords 1) dtype)
    (. (v-hef coords feats 1) shape)
    ; (print (. coords shape))
    ; (print (vhash coords))

    (plt.plot (vhash coords))
    (show)))
