from chess import Board as _board
from chess import Move
from misilelibpy import cls

board = _board()

while board.is_checkmate() is False and board.is_stalemate() is False and board.is_fivefold_repetition() is False and board.is_insufficient_material() is False:
    _cache1 = str(board).split("\n")
    print("".join(f"{i-1} {i2}\n" for i, i2 in zip(range(len(_cache1) + 1, 1, -1), _cache1)), end='')
    print("-----------------")
    print("  a b c d e f g h")
    move = Move.from_uci(input())
    if move not in board.legal_moves:
        print("Illegal")
    else:
        board.push(move)
        cls()

print(board.outcome())
