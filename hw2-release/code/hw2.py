import hw2_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw2_utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (n, d).
        y_train: 1d tensor with shape (n,), whose elememnts are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    n = y_train.size()[0]
    alpha = torch.zeros([n, 1]) # a plain tensor
    alpha.requires_grad_()
    optimizer = optim.SGD([alpha], lr=lr)
    for _ in range(num_iters):
        optimizer.zero_grad()
        K = torch.zeros([n, n])
        for i in range(n):
            for j in range(n):
                K[i][j] = kernel(x_train[i], x_train[j])
        f = 0.5 * (alpha.t() * y_train.view((1, n))) @ K @ (alpha * y_train.view(n, 1)) - torch.ones([1, n]) @ alpha
        f.backward()
        optimizer.step()
        with torch.no_grad():
            alpha[alpha < 0] = 0
            alpha -= lr * alpha.grad
            if c!=None:
                alpha[alpha > c] = c
    return alpha.to(dtype=torch.float32).squeeze()

def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw2_utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (n,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (n, d), denoting the training set.
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (m, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (m,), the outputs of SVM on the test set.
    '''
    n = alpha.size()[0]
    m = x_test.size()[0]
    y_test = torch.zeros([m,])
    for j in range(m):
        K = torch.zeros((n, 1))
        for i in range(n):
            K[i] = kernel(x_test[j], x_train[i])
            y_test[j] = (alpha.view((n, 1)) * y_train.view((n, 1)) * K).sum()
    return y_test   

# x, y = hw2_utils.xor_data()
# print(svm_solver(x, y, 0.1, 100, hw2_utils.poly(2), 0.8))
# alpha_1 = svm_solver(x, y, 0.1, 10000, hw2_utils.poly(2), 0.8)
# alpha_2 = svm_solver(x, y, 0.1, 10000, hw2_utils.rbf(1), 0.8)
# alpha_3 = svm_solver(x, y, 0.1, 10000, hw2_utils.rbf(2), 0.8)
# alpha_4 = svm_solver(x, y, 0.1, 10000, hw2_utils.rbf(3), 0.8)

# def pred_f1(x_test):
#     n = alpha_1.size()[0]
#     m = x_test.size()[0]
#     kernel = hw2_utils.poly(2)
#     y_test = torch.from_numpy(np.zeros([m,]))
#     for j in range(m):
#         K = torch.zeros((n, 1))
#         for i in range(n):
#             K[i] = kernel(x_test[j], x[i])
#             y_test[j] = (alpha_1.view((n, 1)) * y.view((n, 1)) * K).sum()
#     return y_test   

# def pred_f2(x_test):
#     n = alpha_2.size()[0]
#     m = x_test.size()[0]
#     kernel = hw2_utils.rbf(1)
#     y_test = torch.from_numpy(np.zeros([m,]))
#     for j in range(m):
#         K = torch.zeros((n, 1))
#         for i in range(n):
#             K[i] = kernel(x_test[j], x[i])
#             y_test[j] = (alpha_2.view((n, 1)) * y.view((n, 1)) * K).sum()
#     return y_test   

# def pred_f3(x_test):
#     n = alpha_3.size()[0]
#     m = x_test.size()[0]
#     kernel = hw2_utils.rbf(2)
#     y_test = torch.from_numpy(np.zeros([m,]))
#     for j in range(m):
#         K = torch.zeros((n, 1))
#         for i in range(n):
#             K[i] = kernel(x_test[j], x[i])
#             y_test[j] = (alpha_3.view((n, 1)) * y.view((n, 1)) * K).sum()
#     return y_test   

# def pred_f4(x_test):
#     n = alpha_4.size()[0]
#     m = x_test.size()[0]
#     kernel = hw2_utils.rbf(4)
#     y_test = torch.from_numpy(np.zeros([m,]))
#     for j in range(m):
#         K = torch.zeros((n, 1))
#         for i in range(n):
#             K[i] = kernel(x_test[j], x[i])
#             y_test[j] = (alpha_4.view((n, 1)) * y.view((n, 1)) * K).sum()
#     return y_test   

# hw2_utils.svm_contour(pred_f1)
# hw2_utils.svm_contour(pred_f2)
# hw2_utils.svm_contour(pred_f3)
# hw2_utils.svm_contour(pred_f4)
