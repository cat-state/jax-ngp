(import jax
        jax [numpy :as np
             random]
        matplotlib [pyplot :as plt]
        seaborn
        pandas)
(import hash [ngp-hash R3-hash])

(import sdf [show])

(setv KEY (random.PRNGKey 2022))



(setv res
    (lfor nx (range 4 8)
      (do
        (setv 
          [w-key x-key h-key y-key] (random.split KEY 4)
          weights (random.uniform w-key [16 3])
          inv-weights (np.linalg.pinv weights)
          feats (random.uniform h-key [(** 2 20) 16])
          x (random.uniform x-key [(** 10 nx) 3])
          idxs (ngp-hash (np.floor (* x 2048)) (** 2 20))
          f (. feats [idxs])
          tot (np.ones-like feats)
          tot (.add (get tot.at idxs) 1.0)
          overwrites (. tot [idxs])
          y-true (random.uniform y-key [(** 10 nx) 3])

          imprinted (.set 
                        (get feats.at idxs)                     
                        (@ y-true inv-weights))
          f2 (. imprinted [idxs])
          diff (np.median (abs (- y-true (@ f2 weights)))))
        (print (** 10 nx) diff)
        (dict :nx nx :err (cut (.ravel (abs (- y-true (@ f2 weights)))) None 10000)))))
(print res)
(for [r res] (seaborn.kdeplot (:err r) :label (:nx r)))
(plt.legend)
(plt.show)
     

