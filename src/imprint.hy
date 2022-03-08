(import jax
        jax [numpy :as np
             random]
        matplotlib [pyplot :as plt]
        seaborn
        pandas)
(import hash [ngp-hash R3-hash])

(import sdf [show sphere])

(setv KEY (random.PRNGKey 2022))

(defn i** [b x] (int (** 10 x)))

(defn imprint [weights features idxs y]
  (.set (get features.at idxs)
        (@ y (np.linalg.pinv weights))))


(for [nx (np.linspace 4 9 :num 10)]
  (do
    (setv 
      [w-key x-key h-key y-key] (random.split KEY 4)
      weights (random.uniform w-key [16 16])
      feats (random.uniform h-key [(** 2 19) 16])
      x (random.uniform x-key [(i** 10 nx) 3])
      idxs (R3-hash (np.floor (* x 2048)) (** 2 19))
      tot (np.ones-like feats)
      tot (.add (get tot.at idxs) 1.0)
      overwrites (. tot [idxs])
      y-true (sphere x)
      y-true (np.concatenate [y-true (np.zeros [(i** 10 nx) 15])] :axis -1)
      imprinted (imprint weights feats idxs y-true)
      f2 (. imprinted [idxs])
      err (abs (- y-true (@ f2 weights))))
    (print (i** 10 nx) (.mean err))
    (seaborn.kdeplot (cut (.ravel err) None 1000000) :label (i** 10 nx))))

(plt.legend)
(plt.show)
     

