import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()
        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)
        self.normalizer = normalizer_class(embed_dim, affine=True)
        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()


class ResidueBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, kernel_size=3, stride=1, padding=1, downsample=None):
        super(ResidueBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class FeedForward(nn.Sequential):
    def __init__(self, dim_input, dim_hidden, dim_out=None):
        dim_out = dim_input if dim_out is None else dim_out
        super().__init__(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(inplace=False),
            nn.Linear(dim_hidden, dim_out)
        )  # initialise nn.Modules


class Network(nn.Module):
    def __init__(self, planes=4, kernal=3, padding=1, layers=3, n_layers=3, \
                 strides_sizes=[2, 1, 1]):
        self.L_embedder = ResidueBlock(1, planes, kernal, strides_sizes[0], padding, layers)
        self.L_encoder = nn.Sequential(*( \
            ResidueBlock(planes, planes, kernal, strides_sizes[i], padding, layers) \
            for i in range(1, n_layers)))
        self.ff1 = FeedForward(260, 512, dim_out=256)
        self.ff2 = nn.Linear(256, 4)

    def forward(self, x, dlc):
        inputs = self.L_embedder(x)  # [height, width, 4]
        inputs = self.L_encoder(inputs)  # [height/2, width/2, 4]
        inputs = F.interpolate(inputs, size=(16, 16), align_corner=False, mode='Bicubic')  # [16, 16]
        inputs = torch.cat((inputs.flatten(), dlc), dim=0)  # [260]
        inputs = self.ff1(inputs)  # [260]
        inputs = nn.ReLU(inputs)  # [260]
        inputs = self.ff2(inputs)  # [4]
        return inputs
