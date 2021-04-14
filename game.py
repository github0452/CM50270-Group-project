import numpy as np

import settings as s

actions = {0: np.array([1, 0]), 1: np.array([0, 1]), 2: np.array([-1, 0]), 3: np.array([0, -1])}

class Game:
    def reset(self):
        self.players = [np.array([s.MAP_SIZE // 4, s.MAP_SIZE // 2]), np.array([s.MAP_SIZE * 3 // 4, s.MAP_SIZE // 2])]
        self.board = np.zeros((s.MAP_SIZE, s.MAP_SIZE))
        if s.MAP_BORDER:
            self.board[[0, -1],:] = 1
            self.board[:, [0, -1]] = 1
        self.board[tuple(self.players[0])] = -1
        self.board[tuple(self.players[1])] = -2
        self.player_turn = 0
        return self.board, self.players

    def out_bounds(self, cords):
        bx, by = self.board.shape
        return True if cords[0] < 0 or cords[1] < 0 or cords[0] >= bx or cords[1] >= by else False

    def step(self, action):
        player_pos = self.players[self.player_turn]
        self.board[tuple(player_pos)] = 1
        player_pos += actions[action]
        failed = self.out_bounds(player_pos) or self.board[tuple(player_pos)] != 0
        self.board[tuple(player_pos)] = - 1 - self.player_turn
        self.player_turn = (self.player_turn + 1) % 2
        return (self.board, self.players), -10 if failed else 1 ,failed
