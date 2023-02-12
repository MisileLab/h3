from berserk import TokenSession as _session
from berserk import Client as _client
from dotenv import load_dotenv
from os import getenv, _exit
from pprint import pprint
from simple_term_menu import TerminalMenu

load_dotenv()

client = _client(session=_session(getenv("lichess_key")))
account_info = client.account.get()

#pprint(account_info)
print(f"name(id): {account_info['username']}({account_info['id']})")

# custom opening and db

response = ""

while True:
    _res = input()
    if _res == "start":
        _mode = ["1+0 Bullet", "2+1 Bullet", "3+0 Blitz", "3+2 Blitz", "5+0 Blitz", "5+3 Blitz", "10+0 Rapid", "10+5 Rapid", "15+10 Rapid", "30+0 Classical", "30+20 Classical"]
        print("what mode?")
        response = _mode[TerminalMenu(_mode).show()]
        break
    elif _res == "quit":
        _exit(0)
    else:
        print("Invaild")

# for debug
print(response)
