import numpy as np

from Networks.player import Player
from game import Game
import torch
# from gui import GUI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

g = Game()
p = Player(device)
# window = GUI()

g.reset()

while True:
    board, players = g.reset()
    failed = False
    rewards = []
    while not (failed):
        action = p.getAction(board, players[0])
        (board, players), reward, failed = g.step(action)
        rewards.append(reward)
        if not failed:
            (board, players), _, failed = g.step(np.random.randint(0, 4))
        # window.update_frame(board)
    print(rewards)
    p.backward(np.stack(rewards))
