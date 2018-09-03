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
    assert (input.norm() > 0), "Division by zero in second term of equation"
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
    def __init__(self, conv_params, caps_maps=32, caps_dims=8):
        super(PrimaryCapsules, self).__init__()
        self.caps_maps = caps_maps
        # Output of conv2 has 256 (32*8) maps of 6x6.
        # We instead want 32 vectors of 8 dims each.
        self.n_caps = caps_maps * 6 * 6
        self.cap_dims = caps_dims
        self.capsules = nn.ModuleList([
            nn.Conv2d(**conv_params) for _ in range(self.caps_maps)
        ])

    def forward(self, x):
        output = [capsule(x) for capsule in self.capsules]
        output = torch.cat(output)
        print(f"PrimaryCaps: Output size 1: {output.size()}")
        output = output.view(x.size(0), self.caps_maps, self.n_caps,
                             self.cap_dims)
        # Not sure of out3 dims. May be backwards.
        print(f"PrimaryCaps: Output size 2: {output.size()}")
        return squash(output)


class CapsNet(nn.Module):

    def __init__(self, conv1_params, conv2_params):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(**conv1_params)
        self.primary_capsules = PrimaryCapsules(conv2_params)

    def forward(self, x):
        print(f"CapsNet input size", x.size())
        output = self.conv1(x)
        print(f"CapsNet conv1 size", output.size())
        output = self.primary_capsules(output)
        print(f"CapsNet PrimaryCaps size", output.size())
        return output


model = CapsNet(conv1_params, conv2_params)

for batch_idx, (data, target) in enumerate(train_loader):
    test_sample = data
    print(f"Sample size: {test_sample.size()}")
    output = model(data)
    break
