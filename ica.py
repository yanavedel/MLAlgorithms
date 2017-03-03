from numpy import tanh, exp, sqrt
from numpy import max as npmax
from numpy import abs as npabs
from numpy import dot, diag, newaxis, zeros, array
from numpy.random import randn
from scipy.linalg import svd, qr, pinv2


def lc(x, alpha=1.0):
    return tanh(alpha * x)


def lcp(x, alpha=1.0):
    return alpha * (1.0 - tanh(alpha * x) ** 2)


def gauss(x, alpha=1.0):
    return x * exp(-(alpha * x ** 2) / 2.0)


def gaussp(x, alpha=1.0):
    return (1.0 - alpha * x ** 2) * exp(-(alpha * x ** 2) / 2.0)


def cube(x):
    return x ** 3


def cubep(x):
    return 3 * x ** 2


def skew(x):
    return x ** 2


def skewp(x):
    return 2 * x


def rowcenter(X):
    rowmeans = X.mean(axis=-1)
    return rowmeans, X - rowmeans[:, newaxis]


def whiteningmatrix(X, n):
    if (n > X.shape[0]):
        n = X.shape[0]
    U, D, Vt = svd(dot(X, X.T) / X.shape[1], full_matrices=False)
    return dot(diag(1.0 / sqrt(D[0:n])), U[:, 0:n].T), dot(U[:, 0:n], diag(sqrt(D[0:n])))


def decorrelation_gs(w, W, p):
    """
    Gram-schmidt orthogonalization of w against the first p rows of W.
    """
    w = w - (W[0:p, :] * dot(W[0:p, :], w.T)[:, newaxis]).sum(axis=0)
    w = w / sqrt(dot(w, w))
    return w


def decorrelation_witer(W):
    """
    Iterative MDUM decorrelation that avoids matrix inversion.
    """
    lim = 1.0
    tol = 1.0e-05
    W = W / (W ** 2).sum()
    while lim > tol:
        W1 = (3.0 / 2.0) * W - 0.5 * dot(dot(W, W.T), W)
        lim = npmax(npabs(npabs(diag(dot(W1, W.T))) - 1.0))
        W = W1
    return W


def decorrelation_mdum(W):
    U, D, VT = svd(W)
    Y = dot(dot(U, diag(1.0 / D)), U.T)
    return dot(Y, W)


"""FastICA algorithms."""


def ica_def(X, tolerance, g, gprime, orthog, alpha, maxIterations, Winit):
    n, p = X.shape
    W = Winit

    for j in xrange(n):
        w = Winit[j, :]
        it = 1
        lim = tolerance + 1
        while ((lim > tolerance) & (it < maxIterations)):
            wtx = dot(w, X)
            gwtx = g(wtx, alpha)
            g_wtx = gprime(wtx, alpha)
            w1 = (X * gwtx).mean(axis=1) - g_wtx.mean() * w
            w1 = decorrelation_gs(w1, W, j)
            lim = npabs(npabs((w1 * w).sum()) - 1.0)
            w = w1
            it = it + 1
        W[j, :] = w
    return W


def ica_par_fp(X, tolerance, g, gprime, orthog, alpha, maxIterations, Winit):
    n, p = X.shape
    W = orthog(Winit)
    lim = tolerance + 1
    it = 1
    while ((lim > tolerance) and (it < maxIterations)):
        wtx = dot(W, X)
        gwtx = g(wtx, alpha)
        g_wtx = gprime(wtx, alpha)
        W1 = dot(gwtx, X.T) / p - dot(diag(g_wtx.mean(axis=1)), W)
        W1 = orthog(W1)
        lim = npmax(npabs(npabs(diag(dot(W1, W.T))) - 1.0))
        W = W1
        it = it + 1
    return W


def fastica(X, nSources=None, algorithm="parallel fp", decorrelation="mdum", nonlinearity="logcosh", alpha=1.0,
            maxIterations=500, tolerance=1e-05, Winit=None, scaled=True):
    algorithm_funcs = {'parallel fp': ica_par_fp, 'deflation': ica_def}
    orthog_funcs = {'mdum': decorrelation_mdum, 'witer': decorrelation_witer}

    if nonlinearity == 'logcosh':
        g = lc
        gprime = lcp
    elif nonlinearity == 'exp':
        g = gauss
        gprime = gaussp
    elif nonlinearity == 'skew':
        g = skew
        gprime = skewp
    else:
        g = cube
        gprime = cubep

    nmix, nsamp = X.shape

    if nSources is None:
        nSources = nmix
    if Winit is None:
        Winit = randn(nSources, nSources)

    # preprocessing (centering/whitening/pca)
    rowmeansX, X = rowcenter(X)
    Kw, Kd = whiteningmatrix(X, nSources)
    X = dot(Kw, X)

    #kwargs = {'tolerance': tolerance, 'g': g, 'gprime': gprime, 'orthog': orthog_funcs[decorrelation], 'alpha': alpha,
     #         'maxIterations': maxIterations, 'Winit': Winit}
    func = algorithm_funcs[algorithm]

    # run ICA
    W = func(X)#, **kwargs)

    # consruct the sources - means are not restored
    S = dot(W, X)

    # mixing matrix
    A = pinv2(dot(W, Kw))

    if scaled == True:
        S = S / S.std(axis=-1)[:, newaxis]
        A = A * S.std(axis=-1)[newaxis, :]

    return A, W, S