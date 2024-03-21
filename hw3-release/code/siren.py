import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from siren_utils import get_cameraman_tensor, get_coords, model_results


ACTIVATIONS = {
    "relu": torch.relu,
    "sin": torch.sin,
    "tanh": torch.tanh
}

class SingleLayer(nn.Module):
    def __init__(self, in_features, out_features, activation, 
                 bias, is_first):
        super().__init__()
        # TODO: create your single linear layer 
        # with the provided input features, output features, and bias

        # self.torch_activation will contain the appropriate activation function that you should use
        if activation is None:
            self.torch_activation = nn.Identity() # no-op
        elif not activation in ACTIVATIONS:
            raise ValueError("Invalid activation")
        else:
            self.torch_activation = ACTIVATIONS[activation]
        # NOTE: when activation is sin omega is 30.0, otherwise 1.0
        self.omega = 30.0 if activation == "sin" else 1.0

        self.in_features = torch.tensor(in_features)
        self.is_first = is_first
        self.ln = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()


    def init_weights(self):
        with torch.no_grad():
            # TODO: initialize the weights of your linear layer 
            # - First layer params should be initialized in: 
            #     UNIFORM(-1/input_features, 1/input_features)
            # - Every other layer params should be initialized in: 
            #     UNIFORM(-\sqrt{6/input_features}/omega, \sqrt{6/input_features}/omega)
            if self.is_first:
                nn.init.uniform_(self.ln.weight, -1/self.in_features, 1/self.in_features)
            else:
                nn.init.uniform_(self.ln.weight, -torch.sqrt(6/self.in_features)/self.omega, torch.sqrt(6/self.in_features)/self.omega)

    def forward(self, input):
        # TODO: pass the input through your linear layer, multiply by omega, then apply activation
        output = self.torch_activation(self.ln(input) * self.omega)
        return output

# We've implemented the model for you - you need to implement SingleLayer above
# We use 7 hidden_layer and 32 hidden_features in Siren 
#   - you do not need to experiment with different architectures, but you may.
class Siren(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, hidden_layers, activation):
        super().__init__()

        self.net = []
        # first layer
        self.net.append(SingleLayer(in_features, hidden_features, activation,
                                    bias=True, is_first=True))
        # hidden layers
        for i in range(hidden_layers):
            self.net.append(SingleLayer(hidden_features, hidden_features, activation,
                                        bias=True, is_first=False))
        # output layer - NOTE: activation is None
        self.net.append(SingleLayer(hidden_features, out_features, activation=None, 
                                    bias=False, is_first=False))
        # combine as sequential
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        # the input to this model is a batch of (x,y) pixel coordinates
        return self.net(coords)

class MyDataset(Dataset):
    def __init__(self, sidelength) -> None:
        super().__init__()
        self.sidelength = sidelength
        self.cameraman_img = get_cameraman_tensor(sidelength)
        self.coords = get_coords(sidelength)
        # TODO: we recommend printing the shapes of this data (coords and img) 
        #       to get a feel for what you're working with
        print("MyDataset:", self.cameraman_img.size(), self.coords.size())

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        # TODO: return the model input (coords) and output (pixel) corresponding to idx
        return self.coords[idx], self.cameraman_img[idx]
    
def train(total_epochs, batch_size, activation, hidden_size=32, hidden_layer=7):
    # TODO(1): finish the implementation of the MyDataset class
    dataset = MyDataset(sidelength=256)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # TODO(2): implement SingleLayer class which is used by the Siren model
    siren_model = Siren(in_features=2, out_features=1, 
                        hidden_features=hidden_size, hidden_layers=hidden_layer, activation=activation)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    siren_model.to(device)
    print("model device:", next(siren_model.parameters()).device)
    
    # TODO(3): set the learning rate for your optimizer
    learning_rate=0.0001 # 1.0 is usually too large, a common setting is 10^{-k} for k=2,3, or 4
    # TODO: try other optimizers such as torch.optim.SGD
    optim = torch.optim.Adam(lr=learning_rate, params=siren_model.parameters())
    
    # TODO(4): implement the gradient descent train loop
    losses = [] # Track losses to make plot at end
    for epoch in range(total_epochs):
        epoch_loss = 0
        for coords, label in dataloader:
            coords, label = coords.to(device), label.to(device)
            # a. TODO: pass inputs (pixel coords) through model
            model_output = siren_model(coords)
            # b. TODO: compute loss (mean squared error - L2) between:
            #   model outputs (predicted pixel values) and labels (true pixels values)
            loss = nn.functional.mse_loss(model_output, label)

            # loop should end with...
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item() # NOTE: .item() very important!
        epoch_loss /= len(dataloader)
        print(f"Epoch: {epoch}, loss: {epoch_loss:4.5f}", end="\r")
        losses.append(epoch_loss)

    siren_model.to("cpu")

    # example for saving model
    torch.save(siren_model.state_dict(), f"siren_model.p")
    
    # Example code for visualizing results
    # To debug you may want to modify this to be in its own function and use a saved model...
    # You can also save the plots with plt.savefig(path)
    fig, ax = plt.subplots(1, 4, figsize=(16,4))
    model_output, grad, lap = model_results(siren_model)
    ax[0].imshow(model_output, cmap="gray")
    ax[1].imshow(grad, cmap="gray")
    ax[2].imshow(lap, cmap="gray")
    # TODO: in order to really see how your loss is updating you may want to change the axis scale...
    #       ...or skip the first few values
    ax[3].plot(losses)
    # plt.show()
    plt.savefig(f"{activation}.png")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train Siren model.')
    parser.add_argument('-e', '--total_epochs', required=True, type=int)
    parser.add_argument('-b', '--batch_size', required=True, type=int)
    parser.add_argument('-a', '--activation', required=True, choices=ACTIVATIONS.keys())
    args = parser.parse_args()
    
    train(args.total_epochs, args.batch_size, args.activation)