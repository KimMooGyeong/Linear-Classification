import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.linalg as lin

def horizontal_boundary(f, y, epsilon, x_min, x_max):
    x = x_min
    result = []
    if f(x, y) > 0:
        positive = True
    elif f(x, y) <= 0:
        positive = False
    else:
        while f(x, y) == 0:
            #result.append([x, y])
            x += epsilon
            if f(x, y) > 0:
                positive = True
            elif f(x, y) < 0:
                positive = False
    while x <= x_max:
        if f(x, y) > 0 and positive == False:
            result.append([x, y])
            positive = True
        if f(x, y) < 0 and positive == True:
            result.append([x, y])
            positive = False
        if f(x, y) == 0:
            result.append([x,y])
            while f(x, y) == 0 and x <= x_max:
                x += epsilon
                if f(x, y) > 0:
                    positive = True
                elif f(x, y) < 0:
                    positive = False
        x += epsilon
    return result

def D2_boundary(f, epsilon, x_min, x_max, y_min, y_max):
    y = y_min
    result = []
    while y <= y_max:
        result += horizontal_boundary(f, y, epsilon, x_min, x_max)
        y += epsilon
    return np.array(result).reshape((-1, 2))

if __name__ == '__main__':
    mu1 = np.array([0, 0])
    mu2 = np.array([1, 1])
    sigma1 = np.array([[1, 0], [0, 1]])
    sigma2 = np.array([[1, 0], [0, 1]])
    precision1 = lin.inv(sigma1)
    precision2 = lin.inv(sigma2)
    epsilon = 0.05
    x_min = -3
    x_max = 3
    y_min = -3
    y_max = 3

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
    def f(x, y):
        N1 = np.exp(_delta(x, y, mu1, precision1))/(2 * math.pi * lin.det(sigma1))
        N2 = np.exp(_delta(x, y, mu2, precision2))/(2 * math.pi * lin.det(sigma2))
        return N1 - N2
    print(horizontal_boundary(f, 0, epsilon, x_min, x_max))
    X = D2_boundary(f, epsilon, x_min, x_max, y_min, y_max)
    plt.scatter(X[:,0], X[:,1])
    plt.show()
