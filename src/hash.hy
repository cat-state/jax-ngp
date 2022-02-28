(import jax
        jax [numpy :as np]
        functools [partial]) 
        

(require hyrule *)

(defmacro vectorized [_ func-name params signature #* body-forms]
  `(with-decorator (partial np.vectorize :signature ~signature)
     (defn ~func-name ~params ~@body-forms)))


(defn ϕ [d]
  (hy.pyops.reduce (fn [prev _] (** (+ 1 prev) (/ 1 (+ 1 d))))
          (range 0 100) 2.0))


(setv PLASTIC (lfor n-dim (range 0 7)
                  (-> (** (/ 1.0 (ϕ n-dim)) (np.arange 1 (+ 1 n-dim)))
                      (% 1.0))))
(setv
  ; To fit it into an int32
  intfo (np.iinfo np.int32)
  π_2 (np.array 2654435761 :dtype np.uint32)
  π_3 (np.array 805459861 :dtype np.uint32))



(defn quasirandom [n n-dim [seed 0.5]]
  "R_d low discrepancy sequence from
http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/"
  (-> (np.arange 1 (+ n 1))
      (np.expand-dims -1)
      (* (get PLASTIC n-dim))
      (+ seed)
      (% 1.0)))

(vectorized defn R3-hash [x [seed 0.5]] "(3)->()"
  "R_3 hash using R_3 sequence and final mixing from hash-without-sine"
  (setv plastic (-> x 
                 (* (get PLASTIC 3)) (+ seed)))

  (% (* (+ (get plastic 0) (get plastic 1)) 
        (get plastic 2)) 
     1.0))


(import matplotlib.pyplot :as plt)

(vectorized defn ngp-hash [x [seed 0] [T (** 2 14)]] "(3)->()"
  (setv x (.astype x np.uint32)) 
  (-> (^ (get x 0) (^ (* (get x 1) π_2) (* (get x 2) π_3)))
     (% T) 
     (.astype np.float32)
     (/ T)))

(vectorized defn hash-without-sine [x] "(n)->()"
  (setv x (/ x 128))
  (-> x
      (* 0.1031)
      (+ (np.dot x (+ 31.32 (ncut x ::-1))))
      (#%(* (get %1 2) (+ (get %1 1) (get %1 0))))
      (% 1.0)))
 

(do
  (setv fig (plt.figure)
        hashes [ngp-hash R3-hash hash-without-sine])
        
  (for [[i hash] (enumerate hashes)]
   (setv ax (.add-subplot fig 1 (len hashes) (+ 1 i) :projection "3d")
       xyz (np.floor (* (quasirandom 10000 3) 128))
       _ (print (. xyz shape))
       h (hash xyz)
       [x y z] (np.split xyz 3 :axis -1))
     
   (.set-title ax (. hash __name__))
   (.scatter ax x y z :c (/ h 128) :cmap "rainbow"))
  (plt.show))
  

(R3-hash (quasirandom 10 3))

(hy.macroexpand (vectorized defn tst2 [x] "(3)->(1)" (print x) (print None)))
(print tst)




(hash-without-sine (quasirandom 1000 3))

(Rd-hash (np.expand-dims (np.arange 1 10) -1))


