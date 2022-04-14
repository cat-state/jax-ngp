(import jax
        jax [numpy :as np]
        matplotlib [pyplot :as plt]
        functools [partial]
        timeit [timeit])
        

(require hyrule *)

(defn ϕ [d]
  (hy.pyops.reduce (fn [prev _] (** (+ 1 prev) (/ 1 (+ 1 d))))
          (range 0 10000) 2.0))


(setv PLASTIC (lfor n-dim (range 0 7)
                  (-> (** (/ 1.0 (ϕ n-dim)) (np.arange 1 (+ 1 n-dim)))
                      (% 1.0))))
(setv
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

(defn 01->T [x T]
  (.astype (* x T) np.uint32))

(defn summarize [x]
  (* (+ (get x 0) (get x 1)) (get x 2)))


(defmacro vectorized [_ func-name params signature #* body-forms]
  `(with-decorator (partial np.vectorize :signature ~signature)
     (defn ~func-name ~params ~@body-forms)))


; from https://www.shadertoy.com/view/4tXyWN
(vectorized defn iq-hash [x [T (** 2 14)]] "(2),()->()"
  (setv x (.astype x np.uint32)
        q (* 1103515245 (^ (>> x 1) (ncut x ::-1)))
        n (* 1103515245 (^ (get q 0) (>> (get q 1) 3))))
  (% n T))

(vectorized defn ngp-hash2 [x [T (** 2 14)] [seed 0]] "(2),()->()"

  (setv x (.astype x np.uint32)) 
  (-> (^ (get x 0) (* (get x 1) π_2))
     (% T)))

(vectorized defn ngp-hash [x [T (** 2 14)] [seed 0]] "(3),()->()"
  (setv x (.astype x np.uint32)) 
  (-> (^ (get x 0) (^ (* (get x 1) π_2) (* (get x 2) π_3)))
     (% T)))
 

(vectorized defn R3-hash [x [T (** 2 14)] [seed 0.5]] "(3),()->()"
  "R_3 hash using R_3 sequence and final mixing from hash-without-sine"
  (setv plastic (-> x (* (get PLASTIC 3)) (+ seed)))
  (-> (summarize plastic)
     (% 1.0)
     (01->T T)))
 

(vectorized defn R2-hash [x [T (** 2 14)] [seed 0.5]] "(2),()->()"
  "R_3 hash using R_3 sequence and final mixing from hash-without-sine"
  (setv plastic (-> x (* (get PLASTIC 2)) (+ seed)))
  (-> (summarize plastic)
     (% 1.0)
     (01->T T)))

(vectorized defn inv-R3-hash [x [T (** 2 14)] [seed 0.5]] "(3),()->()"
  (setv x (-> x (- seed) (/ (get PLASTIC 3))))
  (-> (summarize x)
      (% 1.0)
      (01->T T)))


(vectorized defn hash-without-sine [x [T (** 2 14)]] "(n),()->()"
  (setv x (/ x 128))
  (-> x
      (* 0.1031)
      (+ (np.dot x (+ 31.32 (ncut x ::-1))))
      (summarize)
      (% 1.0)
      (01->T T)))


(defn benchmark [n]
  (setv hashes (lfor h [ngp-hash R3-hash 
                        hash-without-sine inv-R3-hash] (jax.jit h))
               xyz (quasirandom n 3))
  (for [h hashes]

   (print "benchmarking" n (. h __name__))

   (print f"benchmark {(. h __name__)}"
      (/ (timeit "hash()" "import gc;gc.collect()" :globals (dict :hash (fn [] (jax.block-until-ready (h xyz)))) :number 1000)
         1000))))
(+ 1 1)

;(for [n (lfor n [1e6 1e7 1e8 1e9] (int n))]
;  (benchmark n))


(if (= __name__ "__main__") 
  (do
   (setv fig (plt.figure)
          hashes [ngp-hash R3-hash hash-without-sine inv-R3-hash])
           
   (for [[i hash] (enumerate hashes)]
      (setv ax (.add-subplot fig 1 (len hashes) (+ 1 i) :projection "3d")
          xyz (np.floor (* (quasirandom (** 30 3) 3) 128))
          _ (print (. xyz shape))
          h (hash xyz)
          [x y z] (np.split xyz 3 :axis -1))
         
      (.set-title ax (. hash __name__))
      (.scatter ax x y z :c (/ h 128) :cmap "rainbow"))
   (plt.show)))
