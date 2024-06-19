# from sklearn.metrics import pairwise_distances_argmin
import numpy as np
def pairwise_distances_argmin_manua2(X, Y): # version no optmizada
    n_samples = X.shape[0]
    n_clusters = Y.shape[0]
    indices = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        min_dist = float('inf')
        for j in range(n_clusters):
            dist = np.linalg.norm(X[i] - Y[j])
            if dist < min_dist:
                min_dist = dist
                indices[i] = j
    return indices


'''
algoritmo para el pairwise_distances_argimin:
// Pseudocódigo para pairwise_distances_argmin_manual
Funcion pairwise_distances_argmin_manual(X, Y)
    // versión optimizada con distancia euclidiana
    Distancia <- sqrt(suma((X[:, nuevo_eje] - Y[nuevo_eje, :]) ^ 2, eje=2))
    Retornar arg_min(Distancia, eje=1)
Fin Funcion

// Pseudocódigo para k_means_imp
Funcion k_means_imp(dataset, n_clusters, max_iter = 100, tol = 1e-4)
    
    // 1. Selección de un centroide aleatorio
    rng <- RandomState(42)
    i <- permutacion(dataset.shape[0])[:n_clusters]
    centroides <- dataset[i]

    // 2. Analizo hasta un max_iter. Sirve de límite de precisión
    Para _ desde 0 hasta max_iter - 1
        // 3. Calculo cada punto al centroide más cercano
        etiquetas <- pairwise_distances_argmin_manual(dataset, centroides)

        // 4. Calculo los nuevos centroides
        nuevos_centroides <- array([promedio(dataset[etiquetas == k], eje=0) para k desde 0 hasta n_clusters - 1])

        // 5. Verifico que no se haya excedido la tolerancia
        Si todo(norm(nuevos_centroides - centroides, eje=1) < tol) Entonces
            Romper

        centroides <- nuevos_centroides
    Fin Para
    Retornar centroides, etiquetas
Fin Funcion
'''

def pairwise_distances_argmin_manual(X, Y):
    # version optmizada con eucledian distance
    distancia = np.sqrt(((X[:, np.newaxis] - Y[np.newaxis, :]) ** 2).sum(axis=2))
    return  np.argmin(distancia, axis=1)

def k_means_imp(dataset, n_clusters, max_iter = 100, tol = 1e-4):

    # 1. se selecciona un centroide random
    rng = np.random.RandomState(42)
    i = rng.permutation(dataset.shape[0])[:n_clusters]
    centroides = dataset[i]

    # 2. Analizo hasta un max iter. Sirve de limite de precision
    for _ in range(max_iter):
        # 3. Calculo cada punto al centroide mas cercano
        labels = pairwise_distances_argmin_manual(dataset, centroides)

        # 4. calculo los nuevos centroides
        new_centroides = np.array([dataset[labels==k].mean(0) for k in range(n_clusters)])

        # 5. Verifico que no se haya excedido la tolerancia
        if np.all(np.linalg.norm(new_centroides - centroides, axis=1) < tol):
            break

        centroides = new_centroides
    return centroides, labels