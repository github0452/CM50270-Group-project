
import torch
import numpy as np
from NN import Network
import torch.nn as nn
import torch.optim as optim

dlc = torch.tensor([0,0,0,1])
map = torch.rand((1, 1, 36, 36))

def test():
    net = Network()
    out = net.forward(map, dlc)
    print(out)

dlc = torch.tensor([0,0,0,1])
map = torch.rand((1, 1, 36, 36))
net = Network()
optimiser = optim.Adam(net.parameters(), lr=1e-3)
# self.actor_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=1, gamma=1)
max_g = 1
mse_loss = nn.MSELoss()

t = 0
while(t < 100):
    action_values = net.forward(map, dlc)
    print(action_values)
    action = torch.argmax(action_values)
    reward = 1 if action == 0 else -2
    print(action_values[action], torch.tensor(reward).float())
    loss = mse_loss(action_values[action], torch.tensor(reward).float()) #LOSS FUNCTION

    optimiser.zero_grad()
    loss.backward() # calculate gradient backpropagation
    # torch.nn.utils.clip_grad_norm_(net.parameters(), max_g, norm_type=2) # to prevent gradient expansion, set max
    optimiser.step() # update weights
    # self.actor_scheduler.step()
    print(action, action_values)
    t += 1
