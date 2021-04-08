
import torch
import numpy as np
from Networks.NN import Network
import torch.nn as nn
import torch.optim as optim

class Player:
    def __init__(self):
        self.net = Network()
        self.optimiser = optim.Adam(self.net.parameters(), lr=1e-3)
        self.games_played = 0
        self.action_probs_list = []

    def get_board_dlc(self, state):
        board, x, y = state
        bx, by = board.shape
        def get_cord_val(x, y):
            return torch.tensor(1) if x < 0 or y < 0 or bx >= x or by >= y else board[x, y]
        dlc = torch.stack([get_cord_val(x-1,y), get_cord_val(x+1,y), get_cord_val(x,y-1), get_cord_val(x,y+1)]).flatten()
        return board, dlc

    #forward pass
    def getAction(self, state, sampling=True):
        board, dlc = self.get_board_dlc(state)
        board = torch.tensor(board).unsqueeze(dim=0).unsqueeze(dim=0)
        probs = self.net(board, dlc).unsqueeze(dim=0)
        actions = probs.multinomial(1) if sampling \
            else probs.argmax(dim = 1) # pick an action, actions: torch.Size([100])
        self.action_probs_list.append(probs[[x for x in range(len(probs))], actions])
        return actions.squeeze(dim=0).squeeze(dim=0).item()

    #backward pass
    def backward(self, reward_history):
        # rewards [steps, batch_size]
        flipped_returns = []
        reward_reversed = reward_history[::-1]
        next_return = 0
        for r in reward_reversed:
            next_return = next_return * 0.9 + torch.tensor(r)
            flipped_returns.append(next_return)
        probabilities = torch.stack(self.action_probs_list, 0)
        expected_returns = torch.stack(flipped_returns[::-1], 0)
        # calculate actor loss
        advantage = expected_returns
        logprobabilities = torch.log(probabilities)
        reinforce = (advantage * logprobabilities)
        actor_loss = reinforce.mean()
        #backwards pass
        self.optimiser.zero_grad()
        actor_loss.backward() # calculate gradient backpropagation
        # torch.nn.utils.clip_grad_norm_(net.parameters(), max_g, norm_type=2) # to prevent gradient expansion, set max
        self.optimiser.step() # update weights
        # self.actor_scheduler.step()
        #reset for next backward pass
        self.action_probs_list = []
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
