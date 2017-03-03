import numpy as np


def normalize(x):
    delta = max(x) - min(x)
    m = np.mean(x)
    for i in range(len(x)):
        x[i] = (x[i] - m) / delta



def sigmoid(x, theta):
    z = np.dot(x, theta.T)
    return 1.0 / (1.0 + np.exp(-1.0 * z))


def optim_theta(x, y, theta_0, eps):
    m = len(x)
    flag = True
    theta_prev = np.array(theta_0)
    theta = np.array(theta_0)
    lambd = 0.01

    while flag or np.linalg.norm(theta_prev - theta) >= eps:
        flag = False

        for i in range(len(theta)):
            theta_prev[i] = theta[i]

        for j in range(len(theta_0)):
            for i in range(m):
                if j == 0:
                    theta[j] -= lambd * (1. / m) * (sigmoid(x[i], theta_prev.T) - y[i]) * x[i, j]
                else:
                    theta[j] -= lambd * (1. / m) * (sigmoid(x[i], theta_prev.T) - y[i]) * x[i, j]
                    # - lambd * (1.0 / m) * theta_prev[j]
                    # for regularization

            theta[j] -= lambd * (1.0 / m) * theta_prev[j]

    return theta


def logistic_regr(sample, y, theta_0, x):
    eps = np.sqrt(np.finfo(float).eps)
    m = len(sample)
    n = len(sample[0])
    it = np.ones(shape=(m, n + 1))
    it[:, 1:n + 1] = sample
    theta = optim_theta(it, y, theta_0, eps)
    return [1 if sigmoid(x, theta.T) > 0.5 else 0]

