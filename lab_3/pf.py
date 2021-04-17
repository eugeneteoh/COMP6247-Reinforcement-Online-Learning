import numpy as np
import scipy.stats as st
from tqdm import tqdm
from copy import deepcopy

from numpy.random import default_rng
rng = default_rng(seed=42)

def PF(sig, theta0, Q, R, Ns, resample=False):
    N = sig.shape[0]

    theta = np.zeros((N, Ns, 2))
    error = np.zeros(N)
    weights = np.zeros((N, Ns))
    ess = np.zeros(N)

    # First two estimates are initial guesses
    theta[0] = theta0
    theta[1] = theta0

    weights[:2] = 1/Ns
    
    ess[0] = ESS(weights[0])
    ess[1] = ESS(weights[1])

    for n in tqdm(range(2, N)):
        x_n = np.array([sig[n-1], sig[n-2]])

        weights_sum = 0
        for i in range(Ns):
            theta[n, i] = rng.multivariate_normal(theta[n-1, i], Q)
            likelihood = st.norm.pdf(sig[n], theta[n, i].T @ x_n, R)
            weights[n, i] = weights[n-1, i] * likelihood
            
            weights_sum += weights[n, i]

        weights[n] = weights[n] / weights_sum
        ess[n] = ESS(weights[n])

        theta_mean = np.sum(weights[n][:, np.newaxis] * theta[n], axis=0)
        error[n] = (sig[n] - theta_mean.T @ x_n)**2

        if resample:
            weights[n], theta[n] = sir_resample(weights[n], theta[n])
        
    return theta, weights, error, ess

def sir_resample(weights_n, particles_n):
    Ns = weights_n.shape[0]

    new_weights = np.zeros_like(weights_n)
    new_weights.fill(1/Ns)
    new_particles = np.zeros_like(particles_n)

    cdf = np.cumsum(weights_n)
    u = rng.uniform(0, 1/Ns)
    i = 0
    for j in range(Ns):
        while u > cdf[i]:
            i += 1

        new_particles[j] = particles_n[i]

        u += 1/Ns

    return new_weights, new_particles


def ESS(weights_n):
    return 1. / np.sum(np.square(weights_n))



def KF(sig, theta0, P0, Q, R):
    N = sig.shape[0]
    # Initial conditions
    theta = deepcopy(theta0)
    P = deepcopy(P0)

    x = np.zeros((2, 1))

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

    errorIter = errorIter ** 2


    return thetaIter, errorIter

def EKF_LR(X, y, theta0, P0, Q, R):
    # time length
    N = X.shape[0]

    theta = np.zeros((N, 2))
    P = np.zeros((N, 2, 2))
    error = np.zeros(N)

    # first two estimates are initial guesses
    theta[:2] = theta0
    P[:2] = P0

    for n in range(2, N):
        # predict step
        theta_n_n1 = theta[n-1]
        P_n_n1 = P[n-1] + Q

        # update step
        error[n] = y[n] - sigmoid(theta_n_n1.T @ X[n])
        F_n = sigmoid_jacobian(theta_n_n1, X[n])
        k_n = (P_n_n1 @ F_n.T) / (R + F_n @ P_n_n1 @ F_n.T)

        theta[n] = theta_n_n1 + (k_n * error[n])
        P[n] = (np.eye(2) - k_n @ F_n) @ P_n_n1

    return theta, P, error

def PF_LR(X, y, theta0, Q, R, Ns, resample=False):
    # time length
    N = X.shape[0]

    theta = np.zeros((N, Ns, 2))
    error = np.zeros(N)
    weights = np.zeros((N, Ns))
    ess = np.zeros(N)

    # First two estimates are initial guesses
    theta[:2] = theta0
    weights[:2] = 1/Ns
    ess[0] = ESS(weights[0])
    ess[1] = ESS(weights[1])

    for n in tqdm(range(2, N)):
        weights_sum = 0
        for i in range(Ns):
            theta[n, i] = rng.multivariate_normal(theta[n-1, i], Q)
            likelihood = st.norm.pdf(y[n], sigmoid(theta[n, i].T @ X[n]), R)
            weights[n, i] = weights[n-1, i] * likelihood
            
            weights_sum += weights[n, i]

        weights[n] = weights[n] / weights_sum
        ess[n] = ESS(weights[n])

        theta_mean = np.sum(weights[n][:, np.newaxis] * theta[n], axis=0)
        error[n] = (y[n] - sigmoid(theta_mean.T @ X[n]))

        if resample:
            weights[n], theta[n] = sir_resample(weights[n], theta[n])
        
    return theta, weights, error, ess



def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_jacobian(theta, x_n):
    return sigmoid(theta.T @ x_n) * (1 - sigmoid(theta.T @ x_n)) * x_n


if __name__ == '__main__':
    N = 100

    ex = np.random.randn(N)
    # Second order AR Process with coefficients slowly changing in time
    #

    a0 = np.array([1.2, -0.4])
    A  = np.zeros((N,2))
    omega, alpha = N/2, 0.1
    for n in range(N):
        A[n,0] = a0[0] + (alpha * np.cos(2*np.pi*n/N))
        A[n,1] = a0[1] + (alpha * np.sin(np.pi*n/N))
    S = ex.copy()
    for n in range(2, N):
        x = np.array([S[n-1], S[n-2]])
        S[n] = np.dot(x, A[n,:]) + ex[n]

    theta0 = np.random.rand(2)
    # theta0 = np.array([1.2, -0.4])

    beta = 0.01
    Q = beta * np.eye(2) # process noise
    R = np.var(ex) # observation noise

    Ns = 100

    theta, weights, error = PF(S, theta0, Q, R, Ns, True)