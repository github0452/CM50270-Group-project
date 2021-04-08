import numpy as np

import settings as s


class Game:
    def __init__(self):
        self.actions = {0: np.array([1, 0]), 1: np.array([-1, 0]), 2: np.array([0, 1]), 3: np.array([0, -1])}

    def reset(self):
        self.players = [np.array([s.MAP_SIZE // 4, s.MAP_SIZE // 2]), np.array([s.MAP_SIZE * 3 // 4, s.MAP_SIZE // 2])]
        self.board = np.zeros((s.MAP_SIZE, s.MAP_SIZE))
        self.board[tuple(self.players[0])] = 1
        self.board[tuple(self.players[1])] = 2
        self.player_turn = 0
        return self.board, self.players

    def out_bounds(self, cords):
        bx, by = board.shape()
        return True if cords[0] > 0 and cords[1] > 0 and cords[0] > bx and cords[1] > by else False

    def step(self, action):
        player_pos = self.players[self.player_turn]
        self.board[tuple(player_pos)] += 2
        player_pos += self.actions[action]
        failed = self.out_bounds(player_pos) or self.board[tuple(player_pos)] != 0
        self.board[tuple(player_pos)] = 1 + self.player_turn
        self.player_turn = (self.player_turn + 1) % 2
        return (self.board, self.players), -10 if failed else 0,failed
