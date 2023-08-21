from chess import Board, Move
from berserk import Client, TokenSession
from pathlib import Path

a = Board()
b = Path("token.txt").read_text()
c = Client(session=TokenSession(b))
name = c.account.get()["username"]
me = False
nl = "\n"
while not a.is_game_over():
    gstate = c.board.stream_game_state("ccFnwpDbpPN2")
    me = True
    for i in gstate:
        print(me)
        if i["type"] == "gameFull":
            if i["white"]["id"] != name:
                me = False
                continue
        elif i["type"] == "gameState" and not me:
            try:
                if i["moves"] != "":
                    a.push(Move.from_uci(i["moves"].split(" ")[-1]))
            except AssertionError:
                pass
        print(f"{chr(27)}[2J")
        print(f"""{
            nl.join(f'{8-x2} {x}' for x2, x in enumerate(reversed(str(a).split(nl))))
        }\n  A B C D E F G H""")
        if me:
            while True:
                e = input(">")
                m = Move.from_uci(e)
                print(a.legal_moves)
                if m in a.legal_moves:
                    a.push(m)
                    c.board.make_move("ccFnwpDbpPN2", e)
                    break
                print("inv move")
        print(f"{chr(27)}[2J")
        print(f"""{
            nl.join(f'{8-x2} {x}' for x2, x in enumerate(reversed(str(a).split(nl))))
        }\n  A B C D E F G H""")
        me = not me
