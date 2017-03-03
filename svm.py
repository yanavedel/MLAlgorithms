import math
import numpy as np
from scipy import optimize



def normalize(x):
    delta = max(x) - min(x)
    m = np.mean(x)
    for i in range(len(x)):
        x[i] = (x[i] - m) / delta


def kernel(x, y, sigma=1.):
    norm = np.linalg.norm(x - y) ** 2
    return math.exp(-norm / sigma ** 2)


def kkt_cond(alpha, x, y, func):
    return alpha * (y * func(x) - 1.) == 0


def decision_func(sample, y, alpha, x):
    res = 0.0
    for i in range(len(sample)):
        res += y[i] * alpha[i] * kernel(sample[i], x)
    return res


def func_to_min(x, y, alpha):
    res = 0.0
    res -= np.sum(alpha)
    for i in range(len(x)):
        for j in range(i, len(x)):
            res += 1./2 * y[i] * y[j] * alpha[i] * alpha[j]* kernel(x[i], x[j])
    return res


def func_deriv(x, y, alpha):
    res = []
    for i in range(len(alpha)):
        der = y[i] * y[i] * alpha[i] * kernel(x[i], x[i])
        for j in range(i+1, len(alpha)):
            der += 1. / 2 * y[i] * y[j] * alpha[j] * kernel(x[i], x[j])
        der -= 1.
        res.append(der)
    return res


def constraints(y, i, j, alpha, c):
    k = np.dot(y, alpha) - y[i] * alpha[i] - y[j] * alpha[j]
    constr = ({'type': 'eq',
               'fun' : lambda x, z: y[i] * x + y[j] * z - k,
               'jac' : lambda x: [y[i], y[j]]},)

    constr += ({'type': 'ineq',
                'fun': lambda x: x,
                'jac': lambda x: [1. ,0.]},)

    constr += ({'type': 'ineq',
                'fun': lambda x: c - x,
                'jac': lambda x: [-1., 0.]},)

    constr += ({'type': 'ineq',
                'fun': lambda x: x,
                'jac': lambda x: [0., 1.]},)

    constr += ({'type': 'ineq',
                'fun': lambda x: c - x,
                'jac': lambda x: [0., 1.]},)
    return constr


def smo(func, a0, deriv, eps, x, y, c):
    alpha_prev = np.array(a0)
    alpha_next = np.array(a0)
    flag = True

    while flag or np.linalg.norm(alpha_prev - alpha_next) >= eps:
        flag = False
        f = lambda t: decision_func(x, y, alpha_prev, t)
        kkt_ind = [i for i in range(len(alpha_prev)) if kkt_cond(alpha_prev[i], x[i], y[i], f) == False]
        i = kkt_ind[0]
        if i != 0:
            j = i - 1
        else:
            j = i + 1
        alpha_1 = alpha_prev[i]
        alpha_2 = alpha_prev[j]

        if y[i] != y[j]:
            L = max([0, alpha_2 - alpha_1])
            H = min([c, c + alpha_2 - alpha_1])
        else:
            L = max([0, alpha_2 + alpha_1 - c])
            H = min([c, alpha_2 + alpha_1])

        nu = 2 * np.dot(x[i], x[j])  - np.dot(x[j], x[j]) - np.dot(x[i], x[i])
        E_1 = decision_func(x, y, alpha_prev, x[i])
        E_2 = decision_func(x, y, alpha_prev, x[j])

        alpha_next[j] = alpha_prev[j] - y[j] * (E_1 - E_2) / nu
        if alpha_next[j] > H:
            alpha_next[j] = H
        elif alpha_next[j] < L:
            alpha_next[j] = L

        alpha_next[i] = alpha_prev[i] + y[i] * y[j] * (alpha_prev[j] - alpha_next[j])

        #kkt_ind = kkt_ind[np.isfinite(kkt_ind)]
    return kkt_ind


data = np.loadtxt('ex2data1.txt', delimiter=',')

X = data[:, 0:2]
y = data[:, 2]
#X = normalize(X.T[0])
var = np.transpose(X)
normalize(var[0])
normalize(var[1])

f = lambda t: func_to_min(X, y, t)
der = lambda t: func_deriv(X, y, t)
a0 = [0.0] * len(y)
a0[-1] = 0.5

#res = optimize.minimize(f, a0, jac=der, constraints=constraints(y))#, method='SLSQP', options={'disp': True})
print smo(f, a0, der, 1e-6, X, y)