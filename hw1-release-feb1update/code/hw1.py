import numpy as np
import hw1_utils as utils
import matplotlib.pyplot as plt

def linear_gd(X,Y,lrate=0.1,num_iter=100):
    # return parameters as numpy array
    w = np.zeros([X.shape[1] + 1, 1])
    X_alt_T = np.append(X, np.ones([X.shape[0], X.shape[1]]), 1)
    for i in range(1):
        grad = np.dot(X_alt_T.T, X_alt_T)
        grad = np.dot(grad, w)
        grad -= np.dot(X_alt_T.T, Y.reshape([-1, 1]))
        w -= grad * lrate
    return w.reshape([1, -1])[0]

def linear_normal(X,Y):
    # return parameters as numpy array
    X_alt_T = np.append(X, np.ones([X.shape[0], X.shape[1]]), 1)
    Y_alt = Y.reshape([-1, 1])
    inverse = np.linalg.inv(np.dot(X_alt_T.T, X_alt_T))
    w = np.dot(X_alt_T.T, Y)
    w = np.dot(inverse, w)
    return w.reshape([1, -1])[0]

def plot_linear():
    # return plot
    X, Y = utils.load_reg_data()
    plt.plot(X, Y, 'bo')
    X_t = np.linspace(0, 4, 100)
    w = linear_gd(X, Y)
    Y_t = X_t * w[0] + w[1]
    plt.plot(X_t, Y_t)
    plt.show()
    return []

plot_linear()