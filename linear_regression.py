import numpy as np
from scipy import stats


def covariance(x, y):
    covar = 0.0
    x = np.array(x)
    y = np.array(y)
    for i in range(len(x)):
        covar += (x[i] - np.mean(x)) * (y[i] - np.mean(y)) / len(x)
    return covar


def lin_regr(x, y):
    beta = covariance(x, y) / np.var(x)
    alpha = np.mean(y) - beta * np.mean(x)

    '''
    Or, alternatively

    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return slope, intercept, r_value, p_value, std_err
    '''

    return beta, alpha





