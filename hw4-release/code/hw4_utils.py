import numpy as np
import torch

def generate_data():
    """
    Returns a torch tensor of training data of shape (N,D=2)
    """
    np.random.seed(123)
    n = 400
    X = torch.randn(n, 2)
    #X[:, 1] = 0.0
    (m, M) = (X.min(), X.max())
    X = (X - m) / (M - m) / 4 + 1/8 #aspect ratio preserved
    X[n//2:,:] += 1/2

    return X
