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

(vectorized defn R3-hash [x [seed 0.5]] "(n)->(1)"
  (setv dim (. x shape [-1])
        ; Plastic Sequence based R3->R3 hash
        ; Multiples x with the dim dimensional 
        _ (print dim x.shape)
        plastic (-> x 
                 (* (get PLASTIC 3)) (+ seed) (% 1.0))
        _ (print plastic.shape))
  (np.dot plastic (np.logspace 1 dim)))

(import matplotlib.pyplot :as plt)

(do
  (setv fig (plt.figure)
      ax (.add-subplot fig 1 1 1 :projection "3d")
      xyz (np.floor (* (quasirandom 10000 3) 128))
      _ (print (. xyz shape))
      h (R3-hash xyz)
      [x y z] (np.split h 3 :axis -1))
      
  (.scatter ax x y z :c (/ xyz 128))
  (.show plt)) 
  

(R3-hash (quasirandom 10 3))
(vR3-hash (quasirandom 10 3))
(hy.macroexpand (vectorized defn tst2 [x] "(3)->(1)" (print x) (print None)))
(print tst)


(defn hash-without-sine [x]
  (-> x
      (* 0.1031)
      (% 1.0)
      (+ (np.dot x (+ 31.32
                     (ncut x ::-1))))
      (#%(* (get %1 2) (+ (get %1 1) (get %1 0))))
      (% 1.0)))
      

((np.vectorize hash-without-sine) (np.arange 1 10))

(Rd-hash (np.expand-dims (np.arange 1 10) -1))


