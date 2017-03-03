from scipy import optimize
import numpy as np
import math


def func(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def goldenSection(x, a, b, grad, func):
    phi = (1 + math.sqrt(5)) / 2.
    eps = np.sqrt(np.finfo(float).eps)

    while (b - a) / 2. >= eps:
        x_1 = b - (b - a) / phi
        x_2 = a + (b - a) / phi

        if (func(x - x_1 * grad)) > (func(x - x_2 * grad)):
            a = x_1
            x_1 = x_2
            x_2 = b - (x_1 - a)

        if (func(x - x_1 * grad)) < (func(x - x_2 * grad)):
            b = x_2
            x_2 = x_1
            x_1 = a + (b - x_2)

    return (a + b) / 2.


def gradientDescent(func, x_0, eps, lambd=0.01):
    x_prev = np.array(x_0)
    x_next = np.array(x_0)

    flag = True
    grad_eps = np.sqrt(np.finfo(float).eps)

    while flag or np.linalg.norm(x_prev - x_next) >= eps:
        flag = False
        #print x_next
        x_prev = x_next
        grad = optimize.approx_fprime(x_next, func, [eps, eps])
        print grad
        #lambd = goldenSection(x_next, 0., 0.1, grad, func)
        #print lambd
        # lambd = optimize.minimize_scalar(lambda l: func(x_next - l * grad))
        x_next = x_next - lambd * grad
        #print x_next
    return x_next


x_0 = [5., 6.]
#print gradientDescent(func, x_0, 1e-6)

