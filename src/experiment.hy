(import attrs [define])

(defn hyper []
  (define :frozen True :auto-attribs True :kw-only True))

(with-decorator (hyper)
  (defclass cool []
   (^int x)
   (^str y)))


(print (cool :x 1 :y "cool"))
