white = 1
black = 2
space = 0

import sys

sys.path.append("..")

from config import config
import numpy as np

c = config()


class zobrist:
    def __init__(self) -> None:
        self.table = np.random.randint(1, 10000000, size=(c.height, c.width, 2))
        self.h = 0

    def hash(self, board):
        h = 0
        for i in range(c.height):
            for j in range(c.width):
                if board[i][j] != 0:
                    k = board[i][j]
                    h ^= self.table[i][j][k]
        return h

    def step(self, move, player):
        self.h ^= self.table[move[0]][move[1]][player - 1]
        return self.h

    def destep(self, move, player):
        self.h ^= self.table[move[0]][move[1]][player - 1]
        return self.h
