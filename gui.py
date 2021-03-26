##### Pip install opencv-python
import cv2 as cv
import numpy as np
import settings as _s_
from numpy.lib.stride_tricks import as_strided


__Color__ = {0: (  0,   0,   0),
             1: (  1,   0,   0),
             2: (  0,   1,   0),
             3: (  1, 0.2, 0.2),
             4: (  0.2, 1, 0.2)}  


##https://stackoverflow.com/questions/32846846/quick-way-to-upsample-numpy-array-by-nearest-neighbor-tiling
def tile_array(a, b0, b1):
    r, c = a.shape                                    # number of rows/columns
    rs, cs = a.strides                                # row/column strides 
    x = as_strided(a, (r, b0, c, b1), (rs, 0, cs, 0)) # view a as larger 4D array
    return x.reshape(r*b0, c*b1)                      # create new 2D array

class GUI():
    def __init__(self, _name='untitled'):
        self.buffer = np.random.rand(64, 64, 3)
        self.wname  = _name
        ### define a default window name
        cv.imshow(self.wname, self.buffer)
    
    def update_frame(self, _game_board):
        buffer = np.zeros((_s_.MAP_SIZE, _s_.MAP_SIZE, 3))

        for c in __Color__:
            if c == 0: continue
            buffer[_game_board==c] = __Color__[c]

        ## Nearest Neigbour upscale

        _game_board = tile_array(_game_board, _s_.RENDER_SCALING,
                                              _s_.RENDER_SCALING)
        cv.imshow(self.wname, _game_board)


# def test():
#     g = GUI()
#     for i in range(100000):
#         if i % 1000 == 0:
#             g.update_frame(np.zeros(shape=(64, 64)))
#         print('Post{:10d}'.format(i))


# test()
