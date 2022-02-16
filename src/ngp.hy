(import 
  jax
  jax [numpy :as np]
  matplotlib [use :as mpl-use pyplot :as plt])
(require hyrule *)

(mpl-use "module://matplotlib-backend-kitty")

(plt.plot (np.linspace 0 1 100))

(setv 
  T (** 2 14) 
  π_2 (% 2654435761 (** 2 31))
  π_3 805459861)

(defn spatial-hash [coord]
  (% (^ (get coord 0) 
     (^ (* (get coord 1) π_2)
        (* (get coord 2) π_3))) 
      T))


(setv vhash (jax.vmap spatial-hash))

(defn grid-resolution [level growth-factor coarsest]
  (np.floor 
    (* coarsest (** growth-factor level))))

(defn voxel-idxs [coord level]
  (let [lower (np.floor (* coord (grid-resolution level 2.0)))
        lower (np.expand-dims lower -1)]
    (+ lower (np.array [[0 0 0
                         1 0 0
                         0 1 0
                         0 0 1
                         1 1 0
                         0 1 1
                         1 0 1
                         1 1 1]] :dtype np.int32)))


(defn hash-encoded-features [coord features level]
  (let [voxels (voxel-idxs coord level)
        interpolate-by (- (np.expand-dims coord -1) voxels)
        feature-idxs (vhash voxels)
        voxel-feats (get features feature-idxs)]
   (np.sum (* interpolate-by voxel-feats) :axis 0)))

(setv vhash (jax.vmap spatial-hash)
      coords (-> (np.linspace 0 1000 1000 :dtype np.int32) 
                (np.tile [3 1])
                (.transpose 1 0)))

(print (. coords shape))
(print (vhash coords))

(plt.plot (vhash coords))

(plt.show)
