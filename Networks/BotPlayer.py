from game import Game
import numpy as np

g = Game()

class TronPlayerBOT:
    def get_action(self, raw_state, player_no):
        board, players = raw_state
        actions = {0: np.array([1, 0]), 1: np.array([0, 1]), 2: np.array([-1, 0]), 3: np.array([0, -1])}
        actions_no = [0, 1, 2, 3]
        if player_no == 1:
            loc = players[1]
            board[players[0][0], players[0][1]] = 1.0
        else:
            loc = players[0]
            board[players[1][0], players[1][1]] = 1.0
        # actions.pop(index)
        for action in [0, 1, 2, 3]:
            player_pos = loc
            player_pos += actions[action]
            out_bounds = True if player_pos[0] < 0 or player_pos[1] < 0 or player_pos[0] >= bx or player_pos[1] >= by else False
            if out_bounds or board[tuple(player_pos)] != 0:
                actions_no.pop(action)
        return random.choice(actions_no)


    def train_model(self, n_batch, raw_state, action, next_raw_state, reward, end_game):
        pass

    def load_weights(self):
        pass

    def save_weights(self):
        pass
