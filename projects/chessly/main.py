from chess import Board as _board
from chess import Move, InvalidMoveError
from misilelibpy import cls
from os import _exit

_drawrequest = False
board = _board()

while board.is_game_over(claim_draw=_drawrequest) is False:
    _cache1 = str(board).split("\n")
    print("".join(f"{i-1} {i2}\n" for i, i2 in zip(range(len(_cache1) + 1, 1, -1), _cache1)), end='')
    print("-----------------")
    print("  a b c d e f g h")
    command = input()
    if command == "surrender":
        print("Surrendered")
        _exit(0)
    elif command == "draw":
        if not _drawrequest:
            print("Send draw request")
            _drawrequest = True
        else:
            print("Received draw request")
            break
    else:
        try:
            move = Move.from_uci(command)
        except InvalidMoveError:
            print("No uci string")
        else:
            if move not in board.legal_moves:
                print("Illegal")
            else:
                board.push(move)
                cls()

print(board.outcome(claim_draw=_drawrequest))
