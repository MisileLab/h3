from calendar import monthrange
from datetime import datetime
from tabulate import tabulate
from misilelibpy import cls, read_once, write_once
from os import _exit
from os.path import isfile
from json import dumps, loads

DATA_FILE = "datedata.json"

_help = """
v = view memos
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
        if isfile(DATA_FILE) is False:
            write_once(DATA_FILE, r"[]")
        con = loads(read_once(DATA_FILE))
        self.data = []
        self.data.extend(
            DateDataClass(i["month"], i["day"])
            for i in con
        )

    def save_config(self):
        write_once(DATA_FILE, dumps(self.data))


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
print(tabulate(list_in_list(DateData().data), headers=["month", "day"]), end='')
