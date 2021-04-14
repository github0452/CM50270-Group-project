import time
import numpy as np
from datetime import timedelta

from Networks.ton_player import TronPlayer
from game import Game
import torch
from gui import GUI
import settings as s

g = Game()
p = TronQLearning()
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
            (board, players), reward1, failed1 = g.step(action)
            action = p2.get_action(board, players[1])
            (board, players), reward2, failed2 = g.step(action)
            failed = failed1 or failed2
            p.update_reward(reward1, failed)
            p2.update_reward(reward2, failed)
            window.update_frame(board)
            steps += 1
        if epoch % update_interval == 0: 
            print("Elapsed time:", str(timedelta(seconds=(time.time() - stime) )),
                  "Epoch:", str(epoch) + '/' + str(epochs), "Average game steps:", steps/update_interval)
            stime = time.time()
            steps = 0

def main():
    window_sizes = [9, 15, 25, 35, 45, 55]
    for w in window_sizes:
        ## change window size then train
        s.MAP_SIZE=w
        print('\nTrain with window size:', w)
        ## Tell player to update their view
        p.update_window_size(w)
        p2.update_window_size(w)
        ## Train
        train(epochs=5000)

if __name__ == "__main__":
    main()