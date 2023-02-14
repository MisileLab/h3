# https://github.com/PrismarineJS/minecraft-data/blob/master/data/pc/1.19.3/items.json

from tabulate import tabulate
from datetime import datetime
from simple_term_menu import TerminalMenu
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import InMemoryHistory
from requests import get

from misilelibpy import write_once, read_once
from json import loads, dumps
from os.path import isfile

today = datetime.now()
minecraft_items = {i["name"]: i["stackSize"] for i in get("https://raw.githubusercontent.com/PrismarineJS/minecraft-data/master/data/pc/1.19.3/items.json").json()}
if isfile("ICFdata.json") is False:
    print("please init data")
else:
    data = loads(read_once("ICFdata.json"))

def list_the_values():
    """list the moneybook"""
    print(tabulate(data, headers="keys", tablefmt="rst"))

def init():
    """init and reset the configuration and data"""
    write_once("ICFdata.json", '[]')

def add(amount: int, item: str, borrowed: bool, comment: str, owner: str):
    """add history to the moneybook"""
    data.append({"date": today.strftime("%Y-%m-%d %H:%M:%S"), "value": amount, "comment": comment, "borrowed": borrowed, "item": item, "owner": owner})
    write_once("ICFdata.json", dumps(data))

def remove():
    """remove history to the moneybook"""
    _data = list(map(str, data))
    if not _data:
        print("no data to remove")
        return
    terminal_menu = TerminalMenu(_data, multi_select=True, show_multi_select_hint=True)
    terminal_menu.show()
    for i in terminal_menu.chosen_menu_indices:
        del data[i]
    write_once("ICFdata.json", dumps(data))

if __name__ == "__main__":
    a = input("add/remove/list/init > ")
    if a == "add":
        _history = InMemoryHistory()
        for i in minecraft_items.keys():
            _history.append_string(i)
        session = PromptSession(
            history=_history,
            auto_suggest=AutoSuggestFromHistory(),
            enable_history_search=True
        )
        item = session.prompt("item: ")
        _amount = input("amount(stack is end with s): ")
        try:
            amount = int(_amount)
        except ValueError:
            amount = int(_amount.strip('s')) * minecraft_items[item]
        owner = input("owner: ")
        borrowed = input("borrowed? (y/n): ") == "y"
        comment = input("comment: ")
        add(amount, item, borrowed, comment, owner)
    elif a == "remove":
        remove()
    elif a == "list":
        list_the_values()
    elif a == "init":
        init()
