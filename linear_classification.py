import numpy as np
import numpy.linalg as lin
import distribution as dist

"""linear classification model for 2-dimensional data set.
    if you want to classify higher dimension data,
    it is necessary to modify the basis function phi and variable W"""

# dimension of input variable
D = 2
# dimension of weight vector except the bias
W = 5
# dimension of output varialbe
M = 1

"""Initialize weight vector"""
def init_weight():
    vec_w = 10 * np.random.rand(W+1) - 5 * np.ones(W+1)
    vec_w.reshape((1, W+1))
    return vec_w

"""Define some useful functions"""

# sigmoid function
def _sig(x):
    return np.divide(1, (1 + np.exp(-x)))

# derivative of sigmoid
def _sigp(x):
    return _sig(x) * (1 - _sig(x))

# hyperbolic tangent fundtion
def _tanh(x):
    return 2 * _sig(2*x) - 1

"""***                                                 ***"""
"""basis function should be modified for dffierent problem"""
"""***                                                 ***"""
# basis function
def _phi(x):
    phi0 = 1 # bias
    phi1 = x[0][0]
    phi2 = phi1 ** 2
    phi3 = x[0][1]
    phi4 = phi3 ** 2
    phi5 = phi1 * phi4
    phi = np.array([phi0, phi1, phi2, phi3, phi4, phi5]).reshape((1, W+1))
    return phi

def _phi2(x):
    phi0 = 1 # bias
    phi1 = _tanh(x[0][0])
    phi2 = _tanh(x[0][1])
    phi3 = _tanh(x[0][1] * x[0][0])
    phi4 = _tanh(x[0][0]**2)
    phi5 = _tanh(x[0][1]**2)
    phi = np.array([phi0, phi1, phi2, phi3, phi4, phi5]).reshape((1, W+1))
    return phi

# output function
def output(phi, weight):
    activation = weight @ phi.T
    return _sig(activation)

"""Define error function"""
def _Error(data_set, target, weight, n_samples):
    result = 0
    for n in range(n_sample):
        xn = data_set[n,:]
        yn = output(_phi(xn), weight)
        result += (yn - target[n])**2
    return result / (-2)

"""Define gradient of error function"""
def _grad_E(data_set, target, weight, n_samples):
    result = np.zeros(W+1).reshape((1,W+1))
    for n in range(n_samples):
        xn = data_set[n,:].reshape((1,-1))
        yn = output(_phi(xn), weight)
        result += (yn - target[n]) * _phi(xn)
    return result

"""Define Hessian"""
def _Hessian(data_set, target, weight, n_samples):
    result = np.zeros((W+1)*(W+1)).reshape((W+1, W+1))
    for n in range(n_samples):
        xn = data_set[n,:].reshape((1,-1))
        phin = _phi(xn)
        result += (phin.T) @ phin
    return result

"""One iteration"""
def _iterate(data_set, target, weight, n_samples):
    gradE = _grad_E(data_set, target, weight, n_samples)
    H = _Hessian(data_set, target, weight, n_samples)
    return weight - gradE @ (lin.inv(H))

"""solution method"""
def linear_classify(data_set, target, n_samples, tau):
    weight = init_weight()
    for iter in range(tau):
        weight = _iterate(data_set, target, weight, n_samples)
    return weight

"""Error analysis"""
def error_analysis(data_set, target, weight, n_samples):
    result = 0
    for n in range(n_samples):
        xn = data_set[n,:].reshape((1,-1))
        yn = output(_phi(xn), weight)
        tn = target[n]
        if yn > 0.5:
            yn = 1
        else:
            yn = 0
        result += abs(yn - tn)
    return result * 100 / n_samples

"""function for plotting decision boundary if D = 2"""
def f(x, y, weight):
    vec_x = np.array([x, y]).reshape((1,-1))
    phi = _phi(vec_x).T
    return weight@phi

"""Test data"""
if __name__ == '__main__':
    X_train, T_train, color_dummy, n_samples = dist.gen_data()
    X_val, T_val, color, n_samples = dist.gen_data()
    tau = 100
    weight = linear_classify(X_train, T_train, n_samples, tau)
    result_error = error_analysis(X_val, T_val, weight, n_samples)
    print(result_error)
