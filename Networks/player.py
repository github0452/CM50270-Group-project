
import torch
import numpy as np
from Networks.NN import Network
import torch.nn as nn
import torch.optim as optim

class Player:
    def __init__(self, device):
        self.net = Network().to(device)
        self.optimiser = optim.Adam(self.net.parameters(), lr=1e-3)
        self.action_probs_list = []
        self.device = device

    def get_board_dlc(self, state):
        board, (x, y) = state
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
        # select action
        actions = probs.multinomial(1).squeeze(dim=0) if sampling \
            else probs.argmax(dim = 1) # pick an action, actions: torch.Size([100])
        self.action_probs_list.append(probs[[x for x in range(len(probs))], actions])
        return actions.squeeze(dim=0).item()

    #backward pass
    def backward(self, reward_history):
        # rewards [steps, batch_size]
        flipped_returns = []
        reward_reversed = reward_history[::-1]
        next_return = 0
        for r in reward_reversed:
            next_return = next_return * 0.9 + torch.tensor(r).to(self.device)
            flipped_returns.append(next_return)
        probabilities = torch.stack(self.action_probs_list, 0)
        expected_returns = torch.stack(flipped_returns[::-1], 0)
        # calculate actor loss
        advantage = expected_returns
        logprobabilities = torch.log(probabilities)
        reinforce = (advantage * logprobabilities)
        actor_loss =  - reinforce.mean()
        #backwards pass
        self.optimiser.zero_grad()
        actor_loss.backward() # calculate gradient backpropagation
        self.optimiser.step() # update weights
        #reset for next backward pass
        self.action_probs_list = []
        return actor_loss
