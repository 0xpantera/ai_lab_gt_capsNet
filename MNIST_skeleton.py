import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

torch.manual_seed(4242)

data_dir = "./data/"
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_dir, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.30801,))
                   ])),
    batch_size=64, shuffle=True)

def squash(input):
    """
    Squashing function for a tensor.
    :param input: torch.Tensor
    """
    norm = input.norm()
    squared_norm = norm.pow(2)
    return (squared_norm/(1+squared_norm))*(input/norm)

conv1_params = {
    "in_channels": 1,
    "out_channels": 256,
    "kernel_size": 9,
    "stride": 1
}

conv2_params = {
    "in_channels": 256,
    "out_channels": 256,
    "kernel_size": 9,
    "stride": 2
}


class PrimaryCapsules(nn.Module):
    """
    Primary Capsule Network on MNIST.
    :param conv1_params: Parameters for first Conv2d layer
    :param conv2_params: Parameters for second Conv2d layer
    :param caps_maps: number of feature maps (capsules)
    :param caps_dims: dimension of each capsule's activation vector
    Shape:
        - Input: (batch, channels, height, width)
        - Output: (batch, n_caps, caps_dims)
    """
    def __init__(self, conv1_params, conv2_params, caps_maps=32, caps_dims=8):
        super(PrimaryCapsules, self).__init__()
        self.caps_maps = caps_maps
        # Output of conv2 has 256 (32*8) maps of 6x6.
        # We instead want 32 vectors of 8 dims each.
        self.n_caps = caps_maps * 6 * 6
        self.cap_dims = caps_dims
        self.conv1 = nn.Conv2d(**conv1_params)
        self.conv2 = nn.Conv2d(**conv2_params)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        print(f"Output size 1: {out1.size()}")
        out2 = F.relu(self.conv2(out1))
        print(f"Output size 2: {out2.size()}")
        out3 = out2.view(x.size(0), self.n_caps, self.cap_dims)
        # Not sure of out3 dims. May be backwards.
        print(f"Output size 3: {out3.size()}")
        return squash(out3)


model = PrimaryCapsules(conv1_params, conv2_params)

for batch_idx, (data, target) in enumerate(train_loader):
    test_sample = data
    print(f"Sample size: {test_sample.size()}")
    output = model(data)
    break
