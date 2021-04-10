
## Net Player
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from game import actions
from Networks.TronNet import TronNet

class TronPlayer:
    def __init__(self, model_name='default'):
        super(TronPlayer, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("running on", self.device)
        self.model_name = model_name
        self.net = TronNet().to(self.device)
        
        self.optimiser = optim.Adam(self.net.parameters(), lr=0.001)
        self.action_probs_list = []
        self.action_rewards    = []
        self.eps   = np.finfo(np.float32).eps.item()
        self.epoch = 0
        self.depth = 5
        
    def preprocess(self, _board, _location):
        proximity = np.array([self.depth] * len(actions))
        for i in range(len(actions)):
            for p in range(self.depth):
                if _board[tuple(_location + ((p+1) * actions[i]) )] != 0:
                    proximity[i] = p
                    break
                
        proximity = torch.tensor(proximity).float()
        proximity = proximity.unsqueeze(dim=0)
        _board = torch.tensor(_board).unsqueeze(dim=0).unsqueeze(dim=0)
        return _board.to(self.device).float(), proximity.to(self.device)
    
    def get_action(self, _board, _location):
        board, dlc   = self.preprocess(_board, _location)
        probs, value = self.net(board, dlc)

        m      = Categorical(probs)
        action = m.sample()

        self.action_probs_list.append((m.log_prob(action), value))
        return action.item()

    def update_reward(self, _reward, _end_game):
        self.action_rewards.append(_reward)
    
        if _end_game:
            R = 0
            saved_actions = self.action_probs_list
            policy_losses = []
            values_losses = []
            returns    = []

            for r in self.action_rewards[::-1]:
                R = r + 0.95 * R
                returns.insert(0, R)

            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + 0.0001)

            for (log_prob, value), R in zip(self.action_probs_list, returns):
                advantage = R - value.item()
                policy_losses.append(-log_prob * advantage)
                values_losses.append(F.smooth_l1_loss(value, torch.tensor([[R]]).to(self.device)))

            self.optimiser.zero_grad()
            loss = torch.stack(policy_losses).sum() +\
                   torch.stack(values_losses).sum()
            loss.backward()
            self.optimiser.step()

            self.action_probs_list = []
            self.action_rewards    = []
            self.epoch += 1
            if self.epoch % 1000 == 0:
                self.save_weights(self.model_name)

    def load_weights(self, _filename):
        checkpoint = torch.load(_filename)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        print('Loaded with', self.epoch, 'epochs.')

    def save_weights(self, _filename):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimiser.state_dict(),
        }, _filename)
        print('Model saved.')