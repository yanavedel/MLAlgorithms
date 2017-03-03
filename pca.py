import numpy as np


def pca(data, dim_number=2):
    m, n = data.shape
    data -= data.mean(axis=0)
    #for row in data:
    #    row -= np.mean(row)
    sigma = np.cov(data, rowvar=False)
    u, s, v = np.linalg.svd(sigma)
    u = u[:, :dim_number]

    return np.dot(u.T, data.T)




