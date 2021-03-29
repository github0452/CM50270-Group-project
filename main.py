import numpy as np

from game import Game
from gui import GUI

g = Game()
window = GUI()

g.reset()

while True:
    g.reset()
    failed = False
    while not (failed):
        board, failed = g.step(np.random.randint(0, 4))
        window.update_frame(board)
