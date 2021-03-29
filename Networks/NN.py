class ResidueBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, kernel_size=3, stride=1, padding=1, downsample=None):
        super(ResBlock, self).__init__()
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
    def __init__(self, dim_model, dim_hidden, dim_out=None):
        dim_out = dim_model if dim_out is None else dim_out
        super().__init__(
            nn.Linear(dim_model, dim_hidden),
	        nn.ReLU(inplace = False),
            nn.Linear(dim_hidden, dim_out)
        ) #initialise nn.Modules
        # inputs: [n_batch, n_node, dim_model]
        # outputs: [n_batch, n_node, dim_model]

class Network(nn.Module):
    def __init__(self, inplanes=128, planes=4, kernal=3, padding=1, layers=3, strides_sizes, n_layers, downsample=None):
        self.L_encoder = nn.Sequential(*(ResidueBlock(inplanes, planes, kernal, strides_sizes, padding, layers) for i in range(n_layers)))
        self.ff = FeedForward(input_dim, hidden_dim, 4)

    def forward(self, x):
        inputs = self.L_encoder(x)
        inputs = self.ff(inputs)
        return inputs
