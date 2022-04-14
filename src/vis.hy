(import numpy :as np
        jax.tree-util [tree-map]
        jax.flatten-util [ravel-pytree]
        umap [UMAP plot]
        webdataset [WebDataset WebLoader]
        pickle
        matplotlib [pyplot :as plt]
        sklearn.decomposition [PCA])
(require hyrule [-> ncut "#%"])

(import toolz [curry])

(setv tree-map (curry tree-map))

(setv dataset (-> (WebDataset "./latents_cc12m.tar")
                  (.map-dict :pickle pickle.loads)
                  (.to-tuple "pickle"))
      loader (WebLoader dataset :num-workers 0))


(setv torch2jax (tree-map #%(np.array (.numpy %1)))
      latents (map #%(-> %1 torch2jax ravel-pytree (get 0)) loader)
      latents (np.stack (list latents) :axis 0))
(print latents.shape)
(setv umap (-> (UMAP :n-components 2)
               (.fit latents)))

(do
  (plot.points umap)
  (plt.show))

(do
  (plt.scatter (ncut umap : 0) (ncut umap : 1))
  (plt.show)) 
