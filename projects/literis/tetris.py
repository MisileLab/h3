import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import itertools
from PIL import Image


# Tetris game class
# noinspection PyMethodMayBeStatic
class Tetris:
    """Tetris game class"""

    # BOARD
    MAP_EMPTY = 0
    MAP_BLOCK = 1
    MAP_PLAYER = 2
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20

    TETROMINOS = {
        0: {  # I
            0: [(0, 0), (1, 0), (2, 0), (3, 0)],
            90: [(1, 0), (1, 1), (1, 2), (1, 3)],
            180: [(3, 0), (2, 0), (1, 0), (0, 0)],
            270: [(1, 3), (1, 2), (1, 1), (1, 0)],
        },
        1: {  # T
            0: [(1, 0), (0, 1), (1, 1), (2, 1)],
            90: [(0, 1), (1, 2), (1, 1), (1, 0)],
            180: [(1, 2), (2, 1), (1, 1), (0, 1)],
            270: [(2, 1), (1, 0), (1, 1), (1, 2)],
        },
        2: {  # L
            0: [(1, 0), (1, 1), (1, 2), (2, 2)],
            90: [(0, 1), (1, 1), (2, 1), (2, 0)],
            180: [(1, 2), (1, 1), (1, 0), (0, 0)],
            270: [(2, 1), (1, 1), (0, 1), (0, 2)],
        },
        3: {  # J
            0: [(1, 0), (1, 1), (1, 2), (0, 2)],
            90: [(0, 1), (1, 1), (2, 1), (2, 2)],
            180: [(1, 2), (1, 1), (1, 0), (2, 0)],
            270: [(2, 1), (1, 1), (0, 1), (0, 0)],
        },
        4: {  # Z
            0: [(0, 0), (1, 0), (1, 1), (2, 1)],
            90: [(0, 2), (0, 1), (1, 1), (1, 0)],
            180: [(2, 1), (1, 1), (1, 0), (0, 0)],
            270: [(1, 0), (1, 1), (0, 1), (0, 2)],
        },
        5: {  # S
            0: [(2, 0), (1, 0), (1, 1), (0, 1)],
            90: [(0, 0), (0, 1), (1, 1), (1, 2)],
            180: [(0, 1), (1, 1), (1, 0), (2, 0)],
            270: [(1, 2), (1, 1), (0, 1), (0, 0)],
        },
        6: {  # O
            0: [(1, 0), (2, 0), (1, 1), (2, 1)],
            90: [(1, 0), (2, 0), (1, 1), (2, 1)],
            180: [(1, 0), (2, 0), (1, 1), (2, 1)],
            270: [(1, 0), (2, 0), (1, 1), (2, 1)],
        }
    }

    COLORS = {
        0: (255, 255, 255),
        1: (247, 64, 99),
        2: (0, 167, 247),
    }

    def __init__(self):
        # to avoid warnings just mention the warnings
        self.game_over = False
        self.current_pos = [3, 0]
        self.current_rotation = 0
        self.board = []
        self.bag = []
        self.next_piece = None
        self.score = 0

        self.reset()

    def reset(self):
        """Resets the game, returning the current state"""
        self.board = [[0] * Tetris.BOARD_WIDTH for _ in range(Tetris.BOARD_HEIGHT)]
        self.game_over = False
        self.bag = list(range(len(Tetris.TETROMINOS)))
        random.shuffle(self.bag)
        self.next_piece = self.bag.pop()
        self._new_round(piece_fall=False)
        self.score = 0
        return self._get_board_props(self.board)

    def _get_rotated_piece(self, rotation):
        """Returns the current piece, including rotation"""
        return Tetris.TETROMINOS[self.current_piece][rotation]

    def _get_complete_board(self):
        """Returns the complete board, including the current piece"""
        piece = self._get_rotated_piece(self.current_rotation)
        piece = [np.add(x, self.current_pos) for x in piece]
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y][x] = Tetris.MAP_PLAYER
        return board

    def get_game_score(self):
        """Returns the current game score.

        Each block placed counts as one.
        For lines cleared, it is used BOARD_WIDTH * lines_cleared ^ 2.
        """
        return self.score

    def _new_round(self, piece_fall=False) -> int:
        """Starts a new round (new piece)"""
        score = 0
        if piece_fall:
            # Update board and calculate score
            piece = self._get_rotated_piece(self.current_rotation)
            self.board = self._add_piece_to_board(piece, self.current_pos)
            lines_cleared, self.board = self._clear_lines(self.board)
            score = 1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH
            self.score += score

        # Generate new bag with the pieces
        if len(self.bag) == 0:
            self.bag = list(range(len(Tetris.TETROMINOS)))
            random.shuffle(self.bag)

        self.current_piece = self.next_piece
        self.next_piece = self.bag.pop()
        self.current_pos = [3, 0]
        self.current_rotation = 0

        if not self.is_valid_position(self._get_rotated_piece(self.current_rotation), self.current_pos):
            self.game_over = True
        return score

    def is_valid_position(self, piece, pos):
        """Check if there is a collision between the current piece and the board.
        :returns: True, if the piece position is _invalid_, False, otherwise
        """
        for x, y in piece:
            x += pos[0]
            y += pos[1]
            if x < 0 or x >= Tetris.BOARD_WIDTH \
                    or y < 0 or y >= Tetris.BOARD_HEIGHT \
                    or self.board[y][x] == Tetris.MAP_BLOCK:
                return False
        return True

    def _rotate(self, angle):
        """Change the current rotation"""
        r = self.current_rotation + angle

        if r == 360:
            r = 0
        if r < 0:
            r += 360
        elif r > 360:
            r -= 360

        self.current_rotation = r

    def _add_piece_to_board(self, piece, pos):
        """Place a piece in the board, returning the resulting board"""
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y + pos[1]][x + pos[0]] = Tetris.MAP_BLOCK
        return board

    def _clear_lines(self, board):
        """Clears completed lines in a board"""
        # Check if lines can be cleared
        lines_to_clear = [index for index, row in enumerate(board) if sum(row) == Tetris.BOARD_WIDTH]
        if lines_to_clear:
            board = [row for index, row in enumerate(board) if index not in lines_to_clear]
            # Add new lines at the top
            for _ in lines_to_clear:
                board.insert(0, [0 for _ in range(Tetris.BOARD_WIDTH)])
        return len(lines_to_clear), board

    def _number_of_holes(self, board):
        """Number of holes in the board (empty square with at least one block above it)"""
        holes = 0

        for col in zip(*board):
            tail = itertools.dropwhile(lambda x: x != Tetris.MAP_BLOCK, col)
            holes += len([x for x in tail if x == Tetris.MAP_EMPTY])

        return holes

    def _bumpiness(self, board):
        """Sum of the differences of heights between pair of columns"""
        total_bumpiness = 0
        max_bumpiness = 0
        min_ys = []

        for col in zip(*board):
            tail = itertools.dropwhile(lambda x: x != Tetris.MAP_BLOCK, col)
            n = Tetris.BOARD_HEIGHT - len([x for x in tail])
            min_ys.append(n)

        for (y0, y1) in window(min_ys):
            bumpiness = abs(y0 - y1)
            max_bumpiness = max(bumpiness, max_bumpiness)
            total_bumpiness += bumpiness

        return total_bumpiness, max_bumpiness

    def _height(self, board):
        """Sum and maximum height of the board"""
        sum_height = 0
        max_height = 0
        min_height = Tetris.BOARD_HEIGHT

        for col in zip(*board):
            tail = itertools.dropwhile(lambda x: x != Tetris.MAP_BLOCK, col)
            height = len([x for x in tail])

            sum_height += height
            max_height = max(height, max_height)
            min_height = min(height, min_height)

        return sum_height, max_height, min_height

    def _get_board_props(self, board) -> List[int]:
        """Get properties of the board"""
        lines, board = self._clear_lines(board)
        holes = self._number_of_holes(board)
        total_bumpiness, max_bumpiness = self._bumpiness(board)
        sum_height, max_height, min_height = self._height(board)
        return [lines, holes, total_bumpiness, sum_height]

    def get_next_states(self) -> Dict[Tuple[int, int], List[int]]:
        """Get all possible next states"""
        states = {}
        piece_id = self.current_piece

        if piece_id == 6:
            rotations = [0]
        elif piece_id == 0:
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        # For all rotations
        for rotation in rotations:
            piece = Tetris.TETROMINOS[piece_id][rotation]
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])

            # For all positions
            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                pos = [x, 0]

                # Drop piece
                while self.is_valid_position(piece, pos):
                    pos[1] += 1
                pos[1] -= 1

                # Valid move
                if pos[1] >= 0:
                    board = self._add_piece_to_board(piece, pos)
                    states[(x, rotation)] = self._get_board_props(board)

        return states

    def get_state_size(self):
        """Size of the state"""
        return 4

    def move(self, shift_m, shift_r) -> bool:
        pos = self.current_pos.copy()
        pos[0] += shift_m[0]
        pos[1] += shift_m[1]
        rotation = self.current_rotation
        rotation = (rotation + shift_r + 360) % 360
        piece = self._get_rotated_piece(rotation)
        if self.is_valid_position(piece, pos):
            self.current_pos = pos
            self.current_rotation = rotation
            return True
        return False

    def fall(self) -> bool:
        """:returns: True, if there was a fall move, False otherwise"""
        if not self.move([0, 1], 0):
            # cannot fall further
            # start new round
            self._new_round(piece_fall=True)
            if self.game_over:
                self.score -= 2
        return self.game_over

    def hard_drop(self, pos, rotation, render=False):
        """Makes a hard drop given a position and a rotation, returning the reward and if the game is over"""
        self.current_pos = pos
        self.current_rotation = rotation
        # drop piece
        piece = self._get_rotated_piece(self.current_rotation)
        while self.is_valid_position(piece, self.current_pos):
            if render:
                self.render(wait_key=True)
            self.current_pos[1] += 1
        self.current_pos[1] -= 1
        # start new round
        score = self._new_round(piece_fall=True)
        if self.game_over:
            score -= 2
        if render:
            self.render(wait_key=True)
        return score, self.game_over

    def render(self, wait_key=False):
        """Renders the current board"""
        img = [Tetris.COLORS[p] for row in self._get_complete_board() for p in row]
        img = np.array(img).reshape((Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH, 3)).astype(np.uint8)
        img = img[..., ::-1]  # Convert RRG to BGR (used by cv2)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((Tetris.BOARD_WIDTH * 25, Tetris.BOARD_HEIGHT * 25))
        img = np.array(img)
        cv2.putText(img, str(self.score), (22, 22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.imshow('image', np.array(img))
        if wait_key:
            # this is needed to render during training
            cv2.waitKey(1)


def window(seq, n=2):
    """Returns a sliding window (of width n) over data from the iterable
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
       NB. taken from https://docs.python.org/release/2.3.5/lib/itertools-example.html
    """
    it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
