
import torch
import numpy as np
from NN import Network
def test():
    net = Network()
    dlc = torch.tensor([0,0,0,1])
    map = torch.rand((1, 1, 36, 36))

    out = net.forward(map, dlc)
    print(out)


test()