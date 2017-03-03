import numpy as np
import math


def mean(data):
    return np.mean(data)


def cov_matrix(data):
    return np.cov(data)


def gaussian_discr_analysis(x, y, k, sample):
    n = len(sample)
    x_classes_ind = [np.where(y == i) for i in range(k)]
    x_cl = [x.T[x_classes_ind[i]] for i in range(k)]
    c = []
    for i in range(k):
        c.append(x.T[x_classes_ind[i]])

    m = []
    cov_matrices = []
    x_m = []
    aprior_prob = np.bincount(y) / float(len(y))
    poster_prob = []
    j = 0
    for item in x_cl:
        m.append([mean(item.T[i]) for i in range(len(item.T))])
        cov_matrices.append(cov_matrix(item.T))
        x_m.append([np.array(sample) - m[j]])
        j += 1

    for i in range(k):
        det = np.linalg.det(cov_matrices[i])
        mult = 1. / (((2. *  math.pi) ** (n / 2.)) * math.sqrt(det))
        cov_inv = np.linalg.inv(cov_matrices[i])
        temp = np.dot(x_m[i], cov_inv)
        temp = np.dot(temp, np.transpose(x_m[i]))
        poster_prob.append(mult * math.exp(- (1. / 2) * temp) * aprior_prob[i])

    return np.argmax(poster_prob)



