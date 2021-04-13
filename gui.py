##### Pip install opencv-python
import cv2 as cv
import numpy as np
from numpy.lib.stride_tricks import as_strided

import settings as _s_


def r():
    return np.random.randint(0, 255)


__Color__ = {0: (0, 0, 0),
             1: (r(), r(), r()),
             -1: (r(), r(), r()),
             -2: (r(), r(), r())}

##https://stackoverflow.com/questions/32846846/quick-way-to-upsample-numpy-array-by-nearest-neighbor-tiling
def tile_array(a, b0, b1):
    r, c = a.shape  # number of rows/columns
    rs, cs = a.strides  # row/column strides
    x = as_strided(a, (r, b0, c, b1), (rs, 0, cs, 0))  # view a as larger 4D array
    return x.reshape(r * b0, c * b1)  # create new 2D array

class GUI():
    WINDOW_ID = 0
    @staticmethod
    def get_id():
        GUI.WINDOW_ID+=1
        return GUI.WINDOW_ID

    def __init__(self, _name='untitled'):
        self.buffer = np.zeros((64, 64, 3))
        self.wname  = f'{GUI.get_id():03d}' + ' : ' + _name
        self.inint  = False

        cv.imshow(self.wname, self.buffer)

    def update_frame(self, _game_board):
        if _s_.SHOW_GUI:
            w, h = _game_board.shape
            self.wsize =  w * _s_.RENDER_SCALING
            self.hsize =  h * _s_.RENDER_SCALING

            _game_board = tile_array(_game_board, _s_.RENDER_SCALING,
                                    _s_.RENDER_SCALING)

            self.buffer = np.zeros((self.wsize, self.hsize, 3), np.uint8)

            for c in __Color__:
                if c == 0: continue
                self.buffer[_game_board == c] = __Color__[c]

            cv.imshow(self.wname, self.buffer)
            cv.waitKey(_s_.RENDER_FPS_MS)