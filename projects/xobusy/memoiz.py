from calendar import monthrange
from datetime import datetime
from tabulate import tabulate
from misilelibpy import cls, read_once, write_once
from os import _exit
from os.path import isfile
from json import dumps, loads

CONFIG_FILE = "config.json"

_help = """
v = view calendar
e = edit calendar
q = quit
""".removeprefix("\n")

class DateDataClass:
    def __init__(self, month: int, day: int, name: str, memos: dict):
        self.month = month
        self.day = day
        self.name = name
        self.memos = memos


class DateData:
    def __init__(self):
        if isfile(CONFIG_FILE) is False:
            write_once(CONFIG_FILE, r"[]")
        con = loads(read_once(CONFIG_FILE))
        self.data = []
        for _ in con:
            self.data.append()


def list_in_list(*appender: list):
    a = []
    for i in range(len(appender)):
        if i == 0:
            a.extend([i] for i in appender[i])
        else:
            for i2, i3 in zip(range(len(a)), appender[i]):
                a[i2].append(i3)
    return a

today = datetime.now()
_month = list_in_list(list(range(1, monthrange(today.year, today.month)[1]+1)))

while True:
    _input = False
    print("Memoiz version git")
    cmd = input("> ")
    if cmd == "v":
        print(tabulate(_month, headers="month"), end='')
        _input = True
    elif cmd == "h":
        print(_help, end='')
        _input = True
    elif cmd == "q":
        _exit(0)
    if _input:
        input()
    cls()
