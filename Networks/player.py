
import torch
import numpy as np
from Networks.NN import Network
import torch.nn as nn
import torch.optim as optim

class Player:
    def __init__(self):
        self.net = Network()
        self.games_played = 0
        self.action = []
        self.probs = []

    #forward pass
    def getAction(self, state, sampling=True):
        board, x, y = state
        dlc = torch.tensor(board[x-1:x+1,y-1:y+1].flatten())
        board = torch.tensor(board)
        probs = self.net(board, dlc)
        actions = probs.multinomial(1).squeeze(1) if sampling \
            else probs.argmax(dim = 1) # pick an action, actions: torch.Size([100])
        action_list.append(actions)
        action_probs_list.append(probs[[x for x in range(len(probs))], actions])
        return actions

    #backward pass
    def backward(self, reward_history):
        # rewards [steps, batch_size]
        flipped_returns = []
        reward_reversed = reward_history[::-1]
        next_return = 0
        for r in reward_reversed:
            next_return = next_return * 0.9 + r
            flipped_returns.append(next_return)
        probabilities = torch.stack(self.action_probs_list, 0)
        expected_returns = torch.stack(flipped_returns[::-1], 0)
        # calculate actor loss
        advantage = returns
        logprobabilities = torch.log(probabilities)
        reinforce = (advantage * logprobabilities)
        actor_loss = reinforce.mean()
        #backwards pass
        optimiser.zero_grad()
        actor_loss.backward() # calculate gradient backpropagation
        # torch.nn.utils.clip_grad_norm_(net.parameters(), max_g, norm_type=2) # to prevent gradient expansion, set max
        optimiser.step() # update weights
        # self.actor_scheduler.step()
        return actor_loss


# dlc = torch.tensor([0,0,0,1])
# map = torch.rand((1, 1, 36, 36))
#
# net = Network()
# optimiser = optim.Adam(net.parameters(), lr=1e-3)
# # self.actor_scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=1, gamma=1)
# max_g = 1
# mse_loss = nn.MSELoss()
#
# t = 0
# while(t < 100):
#     action_values = net.forward(map, dlc)
#     print(action_values)
#     action = torch.argmax(action_values)
#     reward = 1 if action == 0 else -2
#     print(action_values[action], torch.tensor(reward).float())
#     loss = mse_loss(action_values[action], torch.tensor(reward).float()) #LOSS FUNCTION
#
#     optimiser.zero_grad()
#     loss.backward() # calculate gradient backpropagation
#     # torch.nn.utils.clip_grad_norm_(net.parameters(), max_g, norm_type=2) # to prevent gradient expansion, set max
#     optimiser.step() # update weights
#     # self.actor_scheduler.step()
#     print(action, action_values)
#     t += 1
