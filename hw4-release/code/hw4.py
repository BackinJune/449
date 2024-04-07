import hw4_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from tqdm import tqdm
# from IPython import embed

class VAE(torch.nn.Module):
    def __init__(self, lam, lrate, latent_dim, loss_fn):
        """
        Initialize the layers of your neural network

        @param lam: Hyperparameter to scale KL-divergence penalty
        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param x - an (N,D) tensor
            @param y - an (N,D) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param latent_dim: The dimension of the latent space

        The network should have the following architecture (in terms of hidden units):
        Encoder Network:
        2 -> 50 -> ReLU -> 50 -> ReLU -> 50 -> ReLU -> (6,6) (mu_layer,logstd2_layer)

        Decoder Network:
        6 -> 50 -> ReLU -> 50 -> ReLU -> 2 -> Sigmoid

        See set_parameters() function for the exact shapes for each weight
        """
        super(VAE, self).__init__()

        self.lrate = lrate
        self.lam = lam
        self.loss_fn = loss_fn
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(2, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU()
        )
        self.to_mu = nn.Linear(50, self.latent_dim)
        self.to_std = nn.Linear(50, self.latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 50), nn.ReLU(), nn.Linear(50, 50), nn.ReLU(), nn.Linear(50, 2), nn.Sigmoid()
        )
        
        self.opt = torch.optim.Adam(
            [{'params': self.encoder.parameters()},
            {'params': self.to_mu.parameters()},
            {'params': self.to_std.parameters()},
            {'params': self.decoder.parameters()}],
            lr = self.lrate
            )

    def set_parameters(self, We1, be1, We2, be2, We3, be3, Wmu, bmu, Wstd, bstd, Wd1, bd1, Wd2, bd2, Wd3, bd3):
        """ Set the parameters of your network

        # Encoder weights:
        @param We1: an (50,2) torch tensor
        @param be1: an (50,) torch tensor
        @param We2: an (50,50) torch tensor
        @param be2: an (50,) torch tensor
        @param We3: an (50,50) torch tensor
        @param be3: an (50,) torch tensor
        @param Wmu: an (6,50) torch tensor
        @param bmu: an (6,) torch tensor
        @param Wstd: an (6,50) torch tensor
        @param bstd: an (6,) torch tensor

        # Decoder weights:
        @param Wd1: an (50,6) torch tensor
        @param bd1: an (50,) torch tensor
        @param Wd2: an (50,50) torch tensor
        @param bd2: an (50,) torch tensor
        @param Wd3: an (2,50) torch tensor
        @param bd3: an (2,) torch tensor

        """
        with torch.no_grad():
            for name, param in self.encoder.named_parameters():
                if name == "0.weight":
                    param.data = We1
                elif name == "0.bias":
                    param.data = be1
                elif name == "2.weight":
                    param.data = We2
                elif name == "2.bias":
                    param.data = be2
                elif name == "4.weight":
                    param.data = We3
                elif name == "4.bias":
                    param.data = be3

            self.to_mu.weight.data = Wmu
            self.to_mu.bias.data = bmu
            self.to_std.weight.data = Wstd
            self.to_std.bias.data = bstd

            for name, param in self.decoder.named_parameters():
                if name == "0.weight":
                    param.data = Wd1
                elif name == "0.bias":
                    param.data = bd1
                elif name == "2.weight":
                    param.data = Wd2
                elif name == "2.bias":
                    param.data = bd2
                elif name == "4.weight":
                    param.data = Wd3
                elif name == "4.bias":
                    param.data = bd3

    def forward(self, x):
        """ A forward pass of your autoencoder

        @param x: an (N, 2) torch tensor

        # return the following in this order from left to right:
        @return y: an (N, 50) torch tensor of output from the encoder network
        @return mean: an (N,latent_dim) torch tensor of output mu layer
        @return stddev_p: an (N,latent_dim) torch tensor of output stddev layer
        @return z: an (N,latent_dim) torch tensor sampled from N(mean,exp(stddev_p/2))
        @return xhat: an (N,D) torch tensor of outputs from f_dec(z)
        """
        # print(x.shape)
        y = self.encoder(x)

        mean = self.to_mu(y)
        stddev_p = self.to_std(y)
        z = mean + torch.randn_like(mean) * torch.exp(stddev_p/2)

        xhat = self.decoder(z)

        return y, mean, stddev_p, z, xhat

    def step(self, x):
        """
        Performs one gradient step through a batch of data x
        @param x: an (N, 2) torch tensor

        # return the following in this order from left to right:
        @return L_rec: float containing the reconstruction loss at this time step
        @return L_kl: kl divergence penalty at this time step
        @return L: total loss at this time step
        """
        y, mean, stddev_p, z, xhat = self.forward(x)
        # embed()
        self.opt.zero_grad()
        L_rec = self.loss_fn(x, xhat)
        # L_kl = - 0.5 * (self.latent_dim + (stddev_p - mean**2 - torch.exp(stddev_p)).sum() / len(x))
        L_kl = - (self.latent_dim + torch.sum(stddev_p - mean**2 - torch.exp(stddev_p), dim=1)) / 2
        L_kl = torch.sum(L_kl) / x.shape[0]
        L = L_rec + self.lam * L_kl
        
        L.backward()
        self.opt.step()
        
        return L_rec, L_kl, L

def fit(net, X, n_iter):
    """ Fit a VAE.  Use the full batch size.
    @param net: the VAE
    @param X: an (N, D) torch tensor
    @param n_iter: int, the number of iterations of training

    # return all of these from left to right:

    @return losses_rec: Array of reconstruction losses at the beginning and after each iteration. Ensure len(losses_rec) == n_iter
    @return losses_kl: Array of KL loss penalties at the beginning and after each iteration. Ensure len(losses_kl) == n_iter
    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return Xhat: an (N,D) NumPy array of approximations to X
    @return gen_samples: an (N,D) NumPy array of N samples generated by the VAE
    """
    # print(X.shape)
    losses_rec = np.zeros(n_iter)
    losses_kl = np.zeros(n_iter)
    losses = np.zeros(n_iter)
    for i in range(n_iter):
        L_rec, L_kl, L = net.step(X)
        losses_rec[i] = L_rec
        losses_kl[i] = L_kl
        losses[i] = L
    _, mean, stddev_p, _, Xhat = net.forward(X)
    z = torch.randn(len(X), net.latent_dim) * torch.exp(stddev_p / 2) + mean
    gen_samples = net.decoder(z)

    return losses_rec, losses_kl, losses, Xhat, gen_samples