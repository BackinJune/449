import numpy as np
import hw1_utils as utils
import matplotlib.pyplot as plt

def linear_gd(X,Y,lrate=0.1,num_iter=100):
    # return parameters as numpy array
    # X: N * d; Y: N,
    w = np.zeros([X.shape[1] + 1, 1])
    X_alt_T = np.append(X, np.ones([X.shape[0], 1]), 1)
    # print(X_alt_T.shape, Y.shape)
    for i in range(num_iter):
        grad = np.dot(X_alt_T.T, X_alt_T)
        grad = np.dot(grad, w)
        grad -= np.dot(X_alt_T.T, Y.reshape([-1, 1]))
        w -= np.log(np.log(np.abs(grad * lrate) + 1) + 1) * 0.01 * np.sign(grad)
    return w.reshape([1, -1])[0]

def linear_normal(X,Y):
    # return parameters as numpy array
    X_alt_T = np.append(X, np.ones([X.shape[0], 1]), 1)
    Y_alt = Y.reshape([-1, 1])
    inverse = np.linalg.pinv(np.dot(X_alt_T.T, X_alt_T))
    w = np.dot(X_alt_T.T, Y_alt)
    w = np.dot(inverse, w)
    return w.reshape([1, -1])[0]

def plot_linear():
    # return plot
    X, Y = utils.load_reg_data()
    print(X.shape)
    ret_plot = plt.subplot()
    ret_plot.plot(X, Y, 'bo')
    w = linear_normal(X, Y)
    Y_t = np.dot(w[:-1], X.T) + w[-1]
    ret_plot.plot(X, Y_t)
    return ret_plot

ret_plot = plot_linear()
plt.show()