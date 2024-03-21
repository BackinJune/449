import skimage
from PIL import Image
import torch
from torchvision.transforms import Resize, Compose, ToTensor, Normalize

def get_coords(sidelength, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelength)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

# load cameraman image from skimage, resize and normalize
def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img.permute(1, 2, 0).view(-1, 1)

# compute the gradient of the output pixels with respect to the input coordinates
def gradient(y, x):
    return torch.autograd.grad(y, [x], grad_outputs=torch.ones_like(y), create_graph=True)[0]

# https://math.stackexchange.com/questions/2498056/what-is-the-difference-between-the-laplacian-and-second-order-derivative
def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div

def model_results(siren_model):
    coords = get_coords(sidelength=256) # model input
    # allows to take derivative w.r.t. input
    coords = coords.clone().detach().requires_grad_(True)
    model_output = siren_model(coords)
    img_grad = gradient(model_output, coords)
    img_laplacian = laplace(model_output, coords)
    
    return (model_output.cpu().view(256,256).detach().numpy(), 
            img_grad.norm(dim=-1).cpu().view(256,256).detach().numpy(),
            img_laplacian.cpu().view(256,256).detach().numpy())