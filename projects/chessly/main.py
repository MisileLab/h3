from chess import Board as _board
from chess import Move

board = _board()

while board.is_checkmate() is False and board.is_stalemate() is False and board.is_fivefold_repetition() is False:
    _cache1 = str(board).split("\n")
    print("".join(f"{i} {i2}\n" for i, i2 in zip(range(len(_cache1) + 1, 1, -1), _cache1)), end='')
    print("-----------------")
    print("  a b c d e f g h")
    move = input()
    if Move.from_uci(move) not in board.legal_moves:
        print("Illegal")
    else:
        board.push(Move.from_uci(move))
    
