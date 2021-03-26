import numpy as np
import settings as s

class Game:
    def __init__(self):
        self.actions = {0: np.array([1, 0]), 1: np.array([-1, 0]), 2: np.array([0, 1]), 3: np.array([0, -1])}
        self.player_turn = 0

    def reset(self):
        self.players = [np.array([s.MAP_SIZE//4, s.MAP_SIZE//2]), np.array([s.MAP_SIZE*3//4, s.MAP_SIZE//2])]
        self.board = np.zeros((s.MAP_SIZE, s.MAP_SIZE))
        self.board[tuple(self.players[0])] = 1
        self.board[tuple(self.players[1])] = 2
        return self.board

    def step(self, action):
        player_pos = self.players[self.player_turn]
        self.board[tuple(player_pos)] += 2
        player_pos += self.actions[action] ## Bug
        self.board[tuple(player_pos)] = 1 + self.player_turn
        failed = self.board[tuple(player_pos)] != 0
        self.player_turn = self.player_turn + 1 % 2
        return self.board, failed

Game()

