import numpy as np
from copy import deepcopy

def kf(sig, theta0, P0, R, beta):
    N = sig.shape[0]
    # Initial conditions
    theta = deepcopy(theta0)
    P = deepcopy(P0)

    x = np.zeros((2, 1))

    Q = beta * np.eye(2)

    thetaIter = np.zeros((2, N))
    errorIter = np.zeros(N)

    # First two estimates are initial guesses
    thetaIter[:, 0] = theta.T
    thetaIter[:, 1] = theta.T

    for n in range(2, N):
        x[0] = sig[n-1]
        x[1] = sig[n-2]

        # theta(n|n-1) = theta(n-1|n-1)
        # P(n|n-1) = P(n-1|n) + Q
        P = P + Q

        # e(n)
        error = sig[n] - x.T @ theta
        # update error plot
        errorIter[n] = error

        # Kalman gain K(n)
        k = (P @ x) / (R + x.T @ P @ x)

        # theta(n|n)
        theta += k @ error
        # update theta plot
        thetaIter[:, n] = theta.T
        # P(n|n)
        P = (np.eye(2) -  k @ x.T) @ P


    return thetaIter, errorIter