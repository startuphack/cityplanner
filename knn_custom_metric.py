import numpy as np
from sklearn.neighbors import NearestNeighbors


def mydist(x, y):
    return np.sum((x - y) ** 4)


X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

nbrs = NearestNeighbors(n_neighbors=4, algorithm='ball_tree', metric=mydist)
nbrs.fit(X)
# NearestNeighbors(algorithm='ball_tree', leaf_size=30, metric='pyfunc',
#                  n_neighbors=4, radius=1.0)
print(nbrs.algorithm)
print(nbrs.kneighbors(X))
