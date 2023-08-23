from chess import Board, Move
from berserk import Client, TokenSession
from pathlib import Path

from time import sleep

b = Path("token.txt").read_text()
c = Client(session=TokenSession(b))
name = c.account.get()["username"]
nl = "\n"
while True:
    a = Board()
    me = False
    whitename = ""
    for x in c.board.stream_incoming_events():
        if x["type"] == "gameStart":
            print(x["game"]["gameId"])
            while not a.is_game_over():
                gstate = c.board.stream_game_state(x["game"]["gameId"])
                me = True
                for i in gstate:
                    print(me)
                    if i["type"] == "gameFull":
                        whitename = i["white"]["id"]
                        if i["white"]["id"] != name:
                            me = False
                            continue
                    elif i["type"] == "gameState" and i["moves"] != "":
                        print(a)
                        a.push(Move.from_uci(i["moves"].split(" ")[-1]))
                    print(f"{chr(27)}[2J")
                    print(f"""{
                        nl.join(
                            f'{8-x2 if whitename == name else x2+1} {x}' 
                            for x2, x in enumerate(reversed(str(a).split(nl)) 
                            if whitename != name else str(a).split(nl))
                        )
                    }\n  A B C D E F G H""")
                    if me:
                        while True:
                            e = input(">")
                            m = Move.from_uci(e)
                            if m in a.legal_moves:
                                c.board.make_move(x["game"]["gameId"], e)
                                break
                            print("inv move")
                    me = not me
                print(f"""{
                nl.join(
                    f'{8-x2 if whitename == name else x2+1} {x}' 
                    for x2, x in enumerate(reversed(str(a).split(nl)) 
                    if whitename != name else str(a).split(nl))
                )
                }\n  A B C D E F G H""")
    sleep(0.1)
