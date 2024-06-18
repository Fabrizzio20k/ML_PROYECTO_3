from sklearn.metrics import pairwise_distances_argmin
import numpy as np

def k_means_imp(dataset, n_clusters, max_iter = 100, tol = 1e-4):

    # 1. se selecciona un centroide random
    rng = np.random.RandomState(42)
    i = rng.permutation(dataset.shape[0])[:n_clusters]
    centroides = dataset[i]

    # 2. Analizo hasta un max iter. Sirve de limite de precision
    for _ in range(max_iter):
        # 3. Calculo cada punto al centroide mas cercano
        labels = pairwise_distances_argmin(dataset, centroides)

        # 4. calculo los nuevos centroides
        new_centroides = np.array([dataset[labels==k].mean(0) for k in range(n_clusters)])

        # 5. Verifico que no se haya excedido la tolerancia
        if np.all(np.linalg.norm(new_centroides - centroides, axis=1) < tol):
            break

        centroides = new_centroides
    return centroides, labels