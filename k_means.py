import numpy as np


def k_mean_upd(centrs, data):
    ds = centrs[:,np.newaxis] - data
    measures = np.sqrt(np.sum(np.square(ds),axis=-1))
    cluster_allocs = np.argmin(measures, axis=0)
    clusters = [data[cluster_allocs == ci] for ci in range(len(centrs))]
    new_centrs = np.asarray([(1 / len(cl)) * np.sum(cl, axis=0) for cl in clusters if len(cl)>0])

    return new_centrs, clusters


def k_mean(centrs, data, n):
    clusters = None
    for ii in range(1, n):
        new_centrs, clusters = k_mean_upd(centrs, data)
        if np.array_equal(centrs,new_centrs):
            break
        else:
            centrs = new_centrs

    return clusters