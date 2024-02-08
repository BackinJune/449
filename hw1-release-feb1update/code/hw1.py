import numpy as np
import hw1_utils as utils
import matplotlib.pyplot as plt

def linear_gd(X,Y,lrate=0.1,num_iter=100):
    # return parameters as numpy array
    # X: N * d; Y: N,
    w = np.zeros([X.shape[1] + 1, 1])
    X = np.c_[np.ones([X.shape[0], 1]), X]
    Y = Y.reshape([-1, 1])
    for i in range(num_iter):
        w -= lrate * (X.T @ X @ w - X.T @ Y) / X.shape[0]
    print(w.shape)
    return w

def linear_normal(X,Y):
    # return parameters as numpy array
    X = np.c_[np.ones([X.shape[0], 1]), X]
    Y = Y.reshape([-1, 1])
    return np.linalg.pinv(X.T @ X) @ X.T @ Y

def plot_linear():
    # return plot
    X, Y = utils.load_reg_data()
    ret_plot = plt.subplot()
    ret_plot.plot(X, Y, 'bo')
    w = linear_normal(X, Y)
    Y_t = X @ w[1:] + w[0]
    print(w)
    ret_plot.plot(X, Y_t)
    return ret_plot

# ret_plot = plot_linear()
# plt.show()
