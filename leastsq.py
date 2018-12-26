import numpy as np
import scipy as sp
from scipy.optimize import leastsq
import matplotlib.pyplot as plt


# target func
def real_func(x):
    return np.sin(2 * np.pi * x)


# polynomial
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)


# unit test
# p = np.poly1d([1,2,3,4,5])
# print(np.poly1d(p))

# loss
def residuals_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret


# ten points
x = np.linspace(0, 1, 10)

# a thousand points
x_points = np.linspace(0, 1, 1000)

# add normal distribution noisy
y_ = real_func(x)
y = [np.random.normal(0, 0.1) + y1 for y1 in y_]

