import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, num_channels):
        """Initialize a building block for ResNet.

        Argument:
            num_channels: the number of channels of the input to Block, and is also
                          the number of channels of conv layers of Block.
        """
        super(Block, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_channels)
        )
        self.final = nn.ReLU()

    def forward(self, x):
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """
        out = self.f(x)
        return self.final(out + x)

class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, num_classes=10):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also 
                          the number of channels of the building block.
            num_classes: the number of output units.
        """
        super(ResNet, self).__init__()
        self.before_view = nn.Sequential(
            nn.Conv2d(1, num_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            Block(num_channels),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.final = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
        out = self.before_view(x)
        out = out.view(out.size(0), out.size(1))
        return self.final(out)