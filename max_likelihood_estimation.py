from gradient_descent import *


def product_func(f, sample, theta):
    res = 1.0
    for i in range(len(sample)):
        res += math.log(f(sample[i], theta))

    return res * (-1.0)


def mle(funct, sample, theta_0):
    f_pr = lambda t: product_func(funct, sample, t)
    theta = gradientDescent(f_pr, theta_0, 1e-6)

    return theta