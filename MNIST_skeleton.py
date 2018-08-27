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


# TODO: implement squash function
def squash(input):
    return input


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
    def __init__(self, caps_maps=32, caps_dims=8,
                 conv1_params, conv2_params):
        super(PrimaryCapsules, self).__init__()
        self.caps_maps = caps_maps
        self.n_caps = caps_maps * 6 * 6
        self.cap_dims = cap_dims
        self.conv1 = nn.Conv2d(**conv1_params)
        self.conv2 = nn.Conv2d(**conv2_params)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        print(f"Output size 1: {out1.size()}")
        out2 = F.relu(self.conv2(out1))
        print(f"Output size 2: {out2.size()}")
        out3 = out2.view(x.size(0), -1, self.cap_dims)
        # Not sure of out3 dims. May be backwards.
        print(f"Output size 3: {out3.size()}")
        return squash(out3)


model = PrimaryCapsules(conv1_params, conv2_params)

for batch_idx, (data, target) in enumerate(train_loader):
    test_sample = data[0, :, :, :]
    print(f"Sample size: {test_sample.size()}")
    output = model(data)
    break
