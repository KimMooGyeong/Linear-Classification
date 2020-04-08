import matplotlib.pyplot as plt
import decision_boundary as disc
import distribution as dist
import linear_classification as lc
import numpy as np

epsilon = 0.01
x_min = -3.0
x_max = 3.0
y_min = -3.0
y_max = 3.0
tau = 100

DBoundary = disc.D2_boundary(dist.f, epsilon, x_min, x_max, y_min, y_max)
X_train, target_train, color_dummy, n_samples = dist.gen_data()
X_val, target_val, color, n_samples = dist.gen_data()
weight = lc.linear_classify(X_train, target_train, n_samples, tau)

def f(x, y):
    return lc.f(x, y, weight)
LCBoundary = disc.D2_boundary(f, epsilon, x_min, x_max, y_min, y_max)

print(lc.error_analysis(X_val, target_val, weight, n_samples), '%')
plt.scatter(X_val[:,0], X_val[:,1], c = color)
plt.scatter(DBoundary[:,0], DBoundary[:,1], c='green', s=0.5)
plt.scatter(LCBoundary[:,0], LCBoundary[:,1], c='black', s=0.5)
plt.show()
