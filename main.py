from gui import GUI
from game import Game

g = Game()
window = GUI()

print(g.reset())

while True:
    g.reset()
    failed = False
    while not(failed):
        board, failed = g.step(np.random.randint(0, 3))
        window.update_frame(board)