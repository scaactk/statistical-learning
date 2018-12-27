import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import leastsq


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


def fitting(M=0):
    """
    :param M: 多项式次数
    :return:
    """

    # 随机初始化 polynomial 参数
    p_init = np.random.rand(M + 1)

    # 最小二乘法
    p_lsq = leastsq(residuals_func, p_init, args=(x, y))
    print("Fitting Parameters:", p_lsq[0])

    # 可视化
    plt.plot(x_points, real_func(x_points), label='real')
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    plt.plot(x, y, 'bo', label='noise')
    plt.legend()
    return p_lsq


p_lsq_0 = fitting(M=1)
