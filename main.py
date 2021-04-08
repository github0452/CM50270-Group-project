import numpy as np

from Networks.player import Player
from game import Game
from gui import GUI

g = Game()
p = Player()
window = GUI()

g.reset()

while True:
    board, players = g.reset()
    failed = False
    rewards = []
    while not (failed):
        (board, players), reward, failed = g.step(p.getAction((board, players[0][0], players[0][1])))
        rewards.append(torch.tensor(reward))
        if not failed:
            (board, players), _, failed = g.step(np.random.randint(0, 4))
        window.update_frame(board)
    p.backward(torch.stack(rewards, 0))
