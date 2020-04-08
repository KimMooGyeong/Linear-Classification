import numpy as np
import numpy.linalg as lin
from  numpy.random import randn
import scipy as sp
import scipy.stats
import math

n_samples = 200

"""Initialize means, covariances, and mix coefficients"""
# mean vectors
mu0 = np.array([0, 0], dtype = float)
mu1 = np.array([1.5, -1.0], dtype = float)
mu2 = np.array([1.1, 1.6], dtype = float)

# covariance symmetric matrices
sigma0 = np.array([1.0, -0.2, -0.2, 0.9], dtype = float).reshape(2,2)
sigma1 = np.array([0.6, 0.6, 0.6, 1.2], dtype = float).reshape(2,2)
sigma2 = np.array([0.5, -0.6, -0.6, 1.0], dtype = float).reshape(2,2)

precision0 = lin.inv(sigma0)
precision1 = lin.inv(sigma1)
precision2 = lin.inv(sigma2)

det_prec0 = lin.det(precision0)
det_prec1 = lin.det(precision1)
det_prec2 = lin.det(precision2)

# mix coeefficients and prior distrubutions
prior = 0.5
etha = 0.6
pi1 = etha
pi2 = 1-etha

"""Obtain data points from each class"""
def gen_data():
    # prior distribution
    prior_dist = sp.stats.binom(n_samples, prior)
    n0_samples = prior_dist.rvs(1)[0]
    n12_samples = n_samples - n0_samples

    pi_dist = sp.stats.binom(n12_samples, etha)
    n1_samples = pi_dist.rvs(1)[0]
    n2_samples = n12_samples - n1_samples

    # target variable and corresponding colors
    target0 = np.array([0] * n0_samples).T
    color0 = np.array(["RED"] * n0_samples).T

    target1 = np.array([1] * n12_samples).T
    color1 = np.array(["Blue"] * n12_samples).T

    color = np.r_[color0, color1]
    target = np.r_[target0, target1]

    # likelihood
    np.random.seed(0)
    X = np.r_[np.dot(randn(n0_samples, 2), sigma0) + mu0, np.dot(randn(n1_samples, 2), sigma1) + mu1,
            np.dot(randn(n2_samples, 2), sigma2) + mu2]
    return X, target, color, n_samples

"""fuctions for newton method"""
def _delta(x, y, mu, precision):
    mux = mu[0]
    muy = mu[1]
    lambdaxx = precision[0][0]
    lambdaxy = precision[0][1]
    lambdayx = precision[1][0]
    lambdayy = precision[1][1]
    term1 = (x - mux) * lambdaxx * (x - mux)
    term2 = (x - mux) * lambdaxy * (y - muy)
    term3 = (y - muy) * lambdayx * (x - mux)
    term4 = (y - muy) * lambdayy * (y - muy)
    return (term1 + term2 + term3 + term4)/(-2)

def _D2Gaussian(x, y, mu, precision, det_prec):
    return (det_prec)*np.exp(_delta(x, y, mu, precision))/(2 * math.pi)

def f(x1, x2):
    term1 = pi1 * _D2Gaussian(x1, x2, mu1, precision1, det_prec1)
    term2 = pi2 * _D2Gaussian(x1, x2, mu2, precision2, det_prec2)
    term3 = _D2Gaussian(x1, x2, mu0, precision0, det_prec0)
    result = term1 + term2 - term3
    return result
