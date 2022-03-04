(import attrs [define asdict astuple]
        jsondiff [diff])
                
        

(defn hyperparams []
  (define :frozen True :auto-attribs True :kw-only True))


(defmacro defhyper [#* cls]
  `(with-decorator (hyperparams)
     (defclass ~@cls)))


(defhyper cool []
  (setv
    ^int x 1
    ^str y "boring"
    ^(get list int) nums [1 2 3]))


(defn maybe-asdict [obj]
  (if (isinstance obj dict)
    obj (asdict obj)))
  

(defn changes [old new]
  (diff (asdict old) (asdict new) :syntax "compact"))
 
(help diff)
(defn summarize [changed])
  
  
  
(changes (cool) (cool :x 5 :nums ["hey" 2 4])) 

(print (cool :x 1 :y "cool"))
(asdict (cool))
