import numpy as np

from Networks.TronPlayer import NetPlayer
from game import Game
import torch
from gui import GUI
import settings as s

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

g = Game()
p = NetPlayer(device)
p2 = NetPlayer(device)
window = GUI()

g.reset()

game_num = 0

while True:
    game_num += 1
    show_gui = True if game_num % s.SHOW_GUI_EVERY == 0 else False
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
        if show_gui:
            window.update_frame(board)
