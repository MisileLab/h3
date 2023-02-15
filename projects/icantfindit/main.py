# https://github.com/PrismarineJS/minecraft-data/blob/master/data/pc/1.19.3/items.json

from tabulate import tabulate
from simple_term_menu import TerminalMenu
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import InMemoryHistory
from pyfiglet import figlet_format
from requests import get
import inquirer

from datetime import datetime
from misilelibpy import write_once, read_once, cls
from json import loads, dumps
from os.path import isfile
from os import _exit
from os import remove as fremove
from subprocess import run
from ast import literal_eval
from validating import *

today = datetime.now()
minecraft_items = {i["name"]: i["stackSize"] for i in get("https://raw.githubusercontent.com/PrismarineJS/minecraft-data/master/data/pc/1.19.3/items.json").json()}
if isfile("ICFdata.json") is False:
    print("please init data")
    _exit(1)
data = loads(read_once("ICFdata.json"))
banner = figlet_format("ICF", font="slant")
_help = """a: add transaction
e: edit transaction
r: remove transaction
l: list transaction
le: list transaction and view with GNU less
i: init data
q: quit console
"""

def list_the_values(less: bool):
    """list transaction"""
    data = [(i["date"], i["owner"], i["item"], i["value"], i["borrowed"], i["comment"])for i in loads(read_once("ICFdata.json"))]
    if not less:
        print(tabulate(data, headers=["date", "owner", "item", "amount", "borrowed", "comment"], tablefmt="rst"))
    else:
        write_once(".tempicf", tabulate(data, headers=["date", "owner", "item", "amount", "borrowed", "comment"], tablefmt="rst"))
        run("less .tempicf", shell=True)
        fremove(".tempicf")

def init():
    """init and reset the configuration and data"""
    write_once("ICFdata.json", '[]')

def add(amount: int, item: str, borrowed: bool, comment: str, owner: str):
    """add transaction"""
    data.append({"date": today.strftime("%Y-%m-%d %H:%M:%S"), "value": amount, "comment": comment, "borrowed": borrowed, "item": item, "owner": owner})
    write_once("ICFdata.json", dumps(data))

def remove():
    """remove transaction"""
    _data = list(map(str, data))
    if not _data:
        print("no data to remove")
        return
    terminal_menu = TerminalMenu(_data, multi_select=True, show_multi_select_hint=True)
    terminal_menu.show()
    for i in terminal_menu.chosen_menu_indices:
        del data[i]
    write_once("ICFdata.json", dumps(data))

def edit():
    """edit transaction"""
    _data = list(map(str, data))
    if not _data:
        print("no data to edit")
        return
    terminal_menu = TerminalMenu(_data)
    _selectid = terminal_menu.show()
    _select = literal_eval(_data[_selectid])
    questions = [
        inquirer.Text("date", message="date of transaction", validate=lambda _, x: validate_datetime_string(x), default=_select["date"]),
        inquirer.Text("owner", message="owner of transaction", default=_select["owner"]),
        inquirer.Text("item", message="item of transaction", validate=lambda _, x: x in minecraft_items, default=_select["item"]),
        inquirer.Text("value", message="item amount of transaction", validate=lambda _, x: validate_int(x), default=_select["value"]),
        inquirer.Confirm("borrowed", message="borrowed or not borrowed transaction", default=_select["borrowed"]),
        inquirer.Text("comment", message="comment of transaction", default=_select["comment"])
    ]
    answer = inquirer.prompt(questions)
    if answer is not None:
        data[_selectid] = answer
        write_once("ICFdata.json", dumps(data))

if __name__ == "__main__":
    while True:
        cls()
        print(banner, end="")
        print("I Can't find it by MisileLaboratory")
        print("Info: use h command if you don't know commands")
        a = input("ICF console > ")
        if a == "a":
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
        elif a == "r":
            remove()
        elif a == "l":
            list_the_values()
        elif a == "le":
            list_the_values(True)
        elif a == "i":
            init()
        elif a == "e":
            edit()
        elif a == "h":
            print(_help, end="")
            input()
        elif a == "q":
            break
