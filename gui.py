##### Pip install opencv-python
import cv2 as cv
import numpy as np

## 0 blank
## 1 Head 3 Body
## 2 Head 4 Body

__Color__ = {0: (  0,   0,   0),
             1: (  1,   0,   0),
             2: (  0,   1,   0),
             3: (  1, 0.2, 0.2),
             4: (  0.2, 1, 0.2)}     

class GUI():
    def __init__(self, _name='untitled'):
        self.buffer = np.random.rand(64, 64, 3)
        self.wname  = _name
        ### define a default window name
        cv.imshow(self.wname, self.buffer)
    
    def update_frame(self, _game_board):
        h, w = _game_board.shape
        buffer = np.zeros((h, w, 3))

        for c in __Color__:
            if c == 0: continue
            buffer[_game_board==c] = __Color__[c]

        cv.imshow(self.wname, _game_board)


# def test():
#     g = GUI()
#     for i in range(100000):
#         if i % 1000 == 0:
#             g.update_frame(np.zeros(shape=(64, 64)))
#         print('Post{:10d}'.format(i))


# test()
