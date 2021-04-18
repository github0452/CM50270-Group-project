import os
import os.path as path
import settings as s
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from game import actions
from Networks.tron_net import TronNet
from gui import GUI

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'is_final'))

class ReplayMem(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capactiy:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, n_batch):
        return random.sample(self.memory, n_batch)

    def __len__(self):
        return len(self.memory)

class TronNet2(nn.Module):
    def __init__(self):
        super(TronNet, self).__init__()
        ## For Board
        self.convs = nn.Sequential(nn.Conv2d(1, 32, 8, stride=4, padding=0), nn.ReLU(),
                                 nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU())
        ## For combined Board
        self.fc_h_v = nn.Linear(3136, 512)
        self.fc_h_a = nn.Linear(3136, 512)
        self.fc_z_v = nn.Linear(512, 1)
        self.fc_z_a = nn.Linear(512, 4)

    def forward(self, board):
        board = self.convs(board).view(-1, 3136)
        v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
        a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
        v, a = v.view(-1, 1, 1), a.view(-1, 4, 1)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams
        q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
        return q

class TronPlayerDQN:
    def __init__(self, model_name='default'):
        super(TronPlayer, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("running on", self.device)
        self.epoch = 0
        self.g = GUI(model_name)
        self.model_name = model_name
        self.view = np.ones((s.MAP_SIZE * 2 - 5, s.MAP_SIZE * 2 - 5))
        self.target_update = 10
        # models
        self.policy_net = TronNet2().to(self.device)
        self.load_weights()
        self.target_net = TronNet2().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.memory = ReplayMemory(10000)
        # optimisers
        self.optimiser = optim.Adam(self.policy_net.parameters(), lr=0.001)

    def update_window_size(self, size):
        self.view = np.ones((size * 2 - 5, size * 2 - 5))

    def preprocess(self, _board, _location):
        self.view[:,:] = 1
        self.view[s.MAP_SIZE - 2 - _location[0]: s.MAP_SIZE * 2 - 4 - _location[0],
                s.MAP_SIZE - 2 - _location[1]: s.MAP_SIZE * 2 - 4 - _location[1]] = _board[1:-1, 1:-1]
        self.g.update_frame(self.view)
        _board = torch.tensor(self.view).unsqueeze(dim=0).unsqueeze(dim=0)
        return _board.to(self.device).float()

    def get_action(self, _board, _location):
        board = self.preprocess(_board, _location)
        probs, value = self.net(board)

        m = Categorical(probs)
        action = m.sample()

        self.action_probs_list.append((m.log_prob(action), value))
        return action.item()

    def train_model(self, n_batch, state, action, next_state, reward, end_game):
        self.memory.push(state, action, next_state, reward, end_game)
        if end_game:
            self.epoch += 1
            just_updated = True
        else:
            just_updated = False
        if len(self.memory) < n_batch:
            return
        transitions = self.memory.sample(n_batch)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_mask = ~torch.cat(batch.is_final).copy() #flip is_final tensor
        non_final_next_states = torch.cat([s for s, is_final in zip(batch.next_state, batch.is_final)
            if is_final is False])
        # pass through network
        pred_v = self.policy_net(s).gather(1, action_batch)
        # calculate actual
        next_v_ = torch.zeros(n_batch, device=self.device)
        next_v_[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        actual_v = (next_v_ * 0.95) + reward_batch
        # Compute Huber loss
        loss = F.smooth_l1_loss(pred_v, actual_v.unsqueeze(1))
        # optimize the model
        self.optimiser.zero_grad()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # update target network if needed
        if just_updated and self.epoch % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def load_weights(self):
        fname = path.join('models', self.model_name)
        if os.path.exists(fname):
            checkpoint = torch.load(fname)
            self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            self.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            print('Loaded with', self.epoch, 'epochs.')
        else:
            print('weights not found for', self.model_name)

    def save_weights(self):
        _filename = path.join('models', self.model_name)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimiser.state_dict(),
        }, _filename)
        print('Model saved.')
