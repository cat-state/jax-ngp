(defmacro "#dbg" [form]
  `(let [x ~form]
     (do
       (print ~(hy.repr form) x)
       x)))
