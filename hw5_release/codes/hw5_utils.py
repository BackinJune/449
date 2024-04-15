import os
import cv2
import gzip
import struct
import array
import numpy as np
from urllib.request import urlretrieve
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset


''' Start GAN helpers '''

BASE_URL = 'http://yann.lecun.com/exdb/mnist/'


# Helper functions and imports
def download(url, filename):
    if not os.path.exists('./data'):
        os.makedirs('./data')
    out_file = os.path.join('./data', filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)


def mnist():
    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(BASE_URL + filename, filename)

    train_images = parse_images('./data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('./data/train-labels-idx1-ubyte.gz')
    test_images  = parse_images('./data/t10k-images-idx3-ubyte.gz')
    test_labels  = parse_labels('./data/t10k-labels-idx1-ubyte.gz')
    return train_images, train_labels, test_images, test_labels


# Load and Prepare Data: Load the MNIST dataset, binarize the images, split into a training dataset 
# of 10000 images and a test set of 10000 images.
def load_mnist():
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:,None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = partial_flatten(train_images) / 255.0
    test_images  = partial_flatten(test_images)  / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]
    train_images = torch.from_numpy(np.round(train_images[0:10000])).float()
    train_labels = torch.from_numpy(train_labels[0:10000]).float()
    test_images = torch.from_numpy(np.round(test_images[0:10000])).float()
    test_labels = torch.from_numpy(test_labels[0:10000])
    return N_data, train_images, train_labels, test_images, test_labels


# Partition the training set into minibatches 
def batch_indices(iter, num_batches, batch_size):
    # iter: iteration index
    # num_batches: number of batches
    # batch_size: batch size
    idx = iter % num_batches
    return slice(idx * batch_size, (idx+1) * batch_size)


# write a function to reshape 784 array into a 28x28 image for plotting
def array_to_image(array):
    return np.reshape(np.array(array), [28, 28])


# concatenate the images for plotting
def concat_images(images, row, col, padding = 3):
    result = np.zeros((28*row+(row-1)*padding,28*col+(col-1)*padding))
    for i in range(row):
        for j in range(col):
            result[i*28+(i*padding):i*28+(i*padding)+28, j*28+(j*padding):j*28+(j*padding)+28] = images[i+j*row]
    return result


class GANDataset(Dataset):
    def __init__(self, data_f, dev, transform=None):
        self.transform = transform
        self._load_data(data_f, dev)

    def _load_data(self, data_f, dev):
        with gzip.open(data_f, 'rb') as fid:
            head = fid.read(16)
            data = fid.read()

        res = struct.unpack(">iiii", head)
        data1 = struct.iter_unpack(">" + "B" * 784, data)

        self.d = torch.zeros(res[1], 1, res[2], res[3])
        for idx, k in enumerate(data1):
            tmp = torch.Tensor(k)
            tmp = tmp.view(1, res[2], res[3])
            if self.transform:
                tmp = self.transform(tmp)
            self.d[idx, :, :, :] = tmp

        self.d = self.d.to(dev)

    def __len__(self):
        return self.d.size()[0]

    def __getitem__(self, idx):
        return self.d[idx, :, :]

''' End GAN helpers '''

''' Start diffusion helpers '''

def create_samples(noise=0.01):
    """
    Create data samples following the pattern "CS446"
    """
    img = np.zeros([64, 128], dtype=np.uint8)
    cv2.putText(img, 'CS446', (0, 31), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 3)
    x, y = np.nonzero(img)
    dist = np.max(x) - np.min(x)
    x = (x - np.mean(x)) / dist
    y = (y - np.mean(y)) / dist
    data = np.stack([y, -x], axis=-1)
    return data + np.random.randn(*data.shape) * noise


def generate_sigmas(sigma_begin, sigma_end, num_levels):
    """
    Generate a list of sigmas in descending order. 
    Sigmas are interpolated between sigma_begin and sigma_end.
    
    Parameters
    ----------
    sigma_begin : float
        The largest sigma, sigma0

    sigma_end : float
        The smallest sigma, sigmaL

    num_levels: int
        Number of sigmas to return, L
    """
    return np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), num_levels))


def plot_score(scorenet, data):
    """
    Plot the predicted score on the grids
    
    Parameters
    ----------
    scorenet : nn.Module
        The neural network for score prediction

    data : np.array, N x 2
        The training data
    """
    plt.figure(figsize=(20, 5))
    x = np.linspace(-2.5, 2.5, 100)
    y = np.linspace(-0.6, 0.6, 50)
    xv, yv = np.meshgrid(x, y)
    samples = np.stack([xv.flatten(), yv.flatten()], axis=1).astype(np.float32)

    scores = (scorenet(torch.from_numpy(samples), 0.1)).detach().numpy()
    scores_norm = np.linalg.norm(scores, axis=-1, ord=2, keepdims=True)
    scores_log1p = scores #/ (scores_norm + 1e-9) * 5# * np.log1p(scores_norm)
    plt.scatter(data[:, 0], data[:, 1])
    plt.quiver(samples[:, 0], samples[:, 1], scores_log1p[:, 0], scores_log1p[:, 1], width=0.002)

    plt.axis('scaled')
    plt.xlim(-2.5, 2.5)
    plt.ylim(-0.6, 0.6)
    plt.title("score function")
    plt.show()


''' End diffusion helpers '''