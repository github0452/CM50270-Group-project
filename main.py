import numpy as np

from Networks.TronPlayer import TronPlayer
from game import Game
import torch
from gui import GUI
import settings as s

g = Game()
p = TronPlayer("default0")
p2 = TronPlayer("default0")
window = GUI()

g.reset()


while True:
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
