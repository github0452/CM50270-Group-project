import numpy as np
import os.path as path
from os import mkdir
import json

class TronQLearning:
    def __init__(self, filename="defaultQ"):
        self.num_steps = 0
        self.epsilon = 0.1
        self.alpha = 0.1
        self.alpha = 0.1
        self.q_table = {}
        self.reward = None
        self.filename = filename
        self.load_weights()

    def get_q(self, key):
        return self.q_table.get(str(key), 0)

    def preprocess(self, _board, _location):
        s = (_board.shape[0]//2, _board.shape[1]//2)
        return np.roll(np.roll(_board, s[0] -_location[0], axis=0),
                       s[1] -_location[1], axis=1)[s[0]-1:s[0]+2,s[1]-1:s[1]+2]

    def best_action(self, _state):
        best_action = np.random.randint(0, 4)
        best_value = self.get_q((_state, best_action))
        for i in range(4):
            value = self.get_q((_state, i))
            if best_value < value:
                best_action = i
                best_value = value
        return best_action, best_value

    def get_action(self, _board, _location):
        if self.reward != None:
            self.num_steps += 1
            self.update_table(self.board)
            # if self.num_steps % 500 == 0:
            #     self.save_weights()
        self.board = self.preprocess(_board, _location)
        if np.random.random() < self.epsilon:
            self.action = np.random.randint(0,4)
        else:
            self.action, _ = self.best_action(str(self.board))
        return self.action

    def update_table(self, _new_state):
        def reflect(state, action):
            axis = 1 if action % 2 == 0 else 0
            return str((str(np.flip(state, axis)), action))

        def rotate(state, action):
            return np.rot90(state), (action + 1) % 4

        state_value = self.get_q((str(self.board), self.action))
        _, next_state_value = self.best_action(str(_new_state))
        new_state_value = state_value + self.alpha * (self.reward + next_state_value - state_value)
        for i in range(4):
            self.board, self.action = rotate(self.board, self.action)
            self.q_table[str((str(self.board), self.action))] = new_state_value
            self.q_table[reflect(self.board, self.action)] = new_state_value

    def update_reward(self, _reward, _end_game):
        self.reward = _reward
        if _end_game:
            self.update_table(np.zeros(1))
            self.reward = None

    def load_weights(self):
        dir = path.dirname(path.realpath("__file__")).replace("\\", "/") + '/models/' + self.filename + ".txt"
        if path.exists(dir):
            with open(dir) as f:
                self.q_table = json.loads(f.read())
                print(("loaded q table size : ", len(self.q_table)))

    def save_weights(self):
        dir = path.dirname(path.realpath("__file__")).replace("\\", "/") + '/models/'
        if not path.exists(dir):
            mkdir(dir)
        with open(dir + self.filename + ".txt", 'w+') as f:
            f.write(json.dumps(self.q_table))
            print(("saved q table size : ", len(self.q_table)))

    def update_window_size(self, w):
        None
