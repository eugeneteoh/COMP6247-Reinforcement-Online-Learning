import numpy as np
import scipy.stats as st
from numba import jit
from tqdm import tqdm

from numpy.random import default_rng
rng = default_rng(seed=42)

@jit(nopython=True)
def sis(sig, theta0, Q, R, Ns):
    N = sig.shape[0]

    theta = np.zeros((N, Ns, 2))
    error = np.zeros(N)
    weights = np.zeros((N, Ns))

    # First two estimates are initial guesses
    theta[0] = theta0
    theta[1] = theta0

    weights[:2] = 1/Ns

    for n in tqdm(range(2, N)):
        x_n = np.array([sig[n-1], sig[n-2]])

        weights_sum = 0
        for i in range(Ns):
            theta[n, i] = rng.multivariate_normal(theta[n-1, i], Q)
            # y_n_i = rng.normal(theta[n, i].T @ x_n, R)
            y_n_i = st.norm.pdf(sig[n], theta[n, i].T @ x_n, R)
            # y_n_i = theta[n, i].T @ x_n
            # y_n_i = rng.normal(sig[n], R)
            # y_n_i = sig[n]
            weights[n, i] = weights[n-1, i] * y_n_i
            
            weights_sum += weights[n, i]

        weights[n] = weights[n] / weights_sum

        theta_mean = np.sum(weights[n][:, np.newaxis] * theta[n], axis=0)
        error[n] = (sig[n] - theta_mean.T @ x_n)**2

    return theta, weights, error