
import torch
import numpy as np
from Networks.NN import Network
import torch.nn as nn
import torch.optim as optim

class Player:
    def __init__(self):
        self.net = Network()
        self.optimiser = optim.Adam(self.net.parameters(), lr=1e-3)
        self.action_probs_list = []

    #forward pass
    def getAction(self, board, player_loc, sampling=True):
        # prep state info
        x, y = player_loc
        board = torch.tensor(board).float() #convert board to tensor
        dlc = torch.tensor([board[x-1,y], board[x+1,y], board[x,y-1], board[x,y+1]])
        # add batch size
        board = board.unsqueeze(dim=0)
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
            next_return = next_return * 0.9 + torch.tensor(r)
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
