import numpy as np
from gui import GUI
from game import Game

g = Game()
window = GUI()

g.reset()

while True:
    g.reset()
    failed = False
    while not(failed):
        board, failed = g.step(np.random.randint(0, 4))
        window.update_frame(board)