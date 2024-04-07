import hw4
import hw4_utils
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def main():

    # initialize parameters
    lr = 0.01
    latent_dim = 6
    lam = 0.00005
    loss_fn = nn.MSELoss()

    # initialize model
    vae = hw4.VAE(lam=lam, lrate=lr, latent_dim=latent_dim, loss_fn=loss_fn)

    # generate data
    X = hw4_utils.generate_data()

    # # Check the number of available GPUs
    # num_gpus = torch.cuda.device_count()
    # if num_gpus > 0:
    #     print("Number of available GPUs:", num_gpus)
    #     for i in range(num_gpus):
    #         print("GPU", i, ":", torch.cuda.get_device_name(i))
    # else:
    #     print("No GPUs available.")

    # if torch.cuda.is_available():
    #     vae = vae.cuda()
    #     X = X.cuda()

    # fit the model to the data
    loss_rec, loss_kl, loss_total, Xhat, gen_samples = hw4.fit(vae, X, n_iter=8000)
    # torch.save(vae.cpu().state_dict(), "vae.pb")
    
    plt.figure()
    plt.plot(loss_rec, label='rec')
    plt.plot(loss_kl, label = 'kl')
    plt.plot(loss_total, label = "empirical risks")
    plt.legend()
    # plt.savefig("../5.3a.png")

    # vae.load_state_dict(torch.load("vae.pb"))
    # _, _, _, _, Xhat = vae.forward(X)

    X = X.cpu().detach().numpy()
    Xhat = Xhat.cpu().detach().numpy()

    plt.figure()
    plt.scatter(X.T[0], X.T[1], color='b', label = 'data points')
    plt.scatter(Xhat.T[0], Xhat.T[1], color='r', label = 'decoded points')
    plt.grid(True)
    plt.legend()
    # # plt.savefig("../5.3b.png")
    
    plt.figure()
    plt.scatter(X.T[0], X.T[1], color='b', label = 'data points')
    plt.scatter(Xhat.T[0], Xhat.T[1], color='r', label = 'decoded points')
    z = torch.rand(len(X), latent_dim)
    gen = vae.decoder(z).cpu().detach().numpy()
    plt.scatter(gen.T[0], gen.T[1], label = 'generated points')
    plt.grid(True)
    plt.legend()
    # plt.savefig("../5.3c.png")

    # plt.show()

if __name__ == "__main__":
    main()
