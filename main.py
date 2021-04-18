import time
import numpy as np
from datetime import timedelta

from Networks.tron_player import TronPlayer
from Networks.tron_q_learning import TronQLearning
from Networks.tron_DQN import TronPlayerDQN
from game import Game
import torch
from gui import GUI
import settings as s

g = Game()
p = TronPlayerDQN()
p2 = TronQLearning()#TronPlayer("default0")
window = GUI()
g.reset()


def train(epochs=5000, update_interval=500):
    stime = time.time()
    steps = 0
    for epoch in range(1, epochs+1):
        board, players = g.reset()
        failed = False
        rewards = []
        rewards2 = []
        while not (failed):
            action = p.get_action(board, players[0])
            (next_board, players), reward1, failed1 = g.step(action)
            action = p2.get_action(board, players[1])
            (board, players), reward2, failed2 = g.step(action)
            failed = failed1 or failed2
            # def train_model(self, n_batch, state, action, next_state, reward, end_game):
            p.train_model(reward1, failed)
            p2.update_reward(reward2, failed)
            window.update_frame(board)
            steps += 1
        if epoch % update_interval == 0:
            print("Elapsed time: {0} Epoch: {1}/{2} Average game steps: {3}".format(str(timedelta(seconds=(time.time() - stime) )), str(epoch), str(epochs), steps/update_interval))
            p.save_weights()
            p2.save_weights()
            stime = time.time()
            steps = 0

def main():
    window_sizes = [7, 9, 15, 25, 35, 45, 55]
    for w in window_sizes:
        ## change window size then train
        s.MAP_SIZE=w
        print('\nTrain with window size:', w)
        ## Tell player to update their view
        p.update_window_size(w)
        p2.update_window_size(w)
        ## Train
        train(epochs=10000, update_interval=1000)

if __name__ == "__main__":
    main()
