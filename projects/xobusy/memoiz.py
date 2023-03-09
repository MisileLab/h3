from calendar import monthrange
from datetime import datetime
from tabulate import tabulate
from simple_term_menu import TerminalMenu
from subprocess import run
from misilelibpy import read_once, write_once
from os.path import isfile
from os import remove, listdir, _exit
from json import dumps, loads

today = datetime.now()
_month = today.month
_year = today.year
DATA_FILE = f"datedata_{_year}"
NOTE_FOLDER = "notes"

class DateDataClass:
    def __init__(self, day: int, memos: list):
        self.day = day
        self.memos = []
        self.memos.extend(Memo(i["name"], i["study"]) for i in memos)

    def json(self):
        _memos = [i.json() for i in self.memos]
        return {
            "day": self.day,
            "memos": _memos
        }

    def names(self):
        return [i.name for i in self.memos]

class Memo:
    def __init__(self, name: str, study: str):
        self.name = name
        self.study = study

    def json(self):
        return { "name": self.name, "study": self.study }

    def __repr__(self):
        return f"{str(self.__class__)}: {str(self.__dict__)}"

    def __eq__(self, other):
        return self.name == other.name and self.study == other.study

class DateData:
    def __init__(self, month: int):
        self.load_config(month)
        self.detect_memos()

    def save_config(self):
        write_once(f"{DATA_FILE}_{self.month}.json", dumps([i.json() for i in self.data]))

    def load_config(self, month: int):
        self.month = month
        if isfile(f"{DATA_FILE}_{self.month}.json") is False:
            write_once(f"{DATA_FILE}_{self.month}.json", r"[]")
        con = loads(read_once(f"{DATA_FILE}_{self.month}.json"))
        self.data = []
        self.data.extend(
            DateDataClass(i["day"], i["memos"])
            for i in con
        )
        if len(self.data)-monthrange(_year, month)[1] < 0:
            self.data.extend(
                DateDataClass(i, [])
                for i in range(1, monthrange(_year, month)[1] - len(self.data)+1)
            )

    def detect_memos(self):
        for i in listdir(NOTE_FOLDER):
            myear, mmonth, mday, mname = map(str, i.strip(".md").split("-", 3))
            myear = int(myear)
            mmonth = int(mmonth)
            mday = int(mday)
            if myear == _year and mmonth == self.month and Memo(i, mname) not in self.data[mday-1].memos:
                self.data[mday-1].memos.append(Memo(i, mname))

    def length_memos(self):
        return [len(i.memos) for i in self.data]


def list_in_list(*appender: list):
    # sourcery skip: instance-method-first-arg-name
    a = []
    for i in range(len(appender)):
        if i == 0:
            a.extend([i] for i in appender[i])
        else:
            for i2, i3 in zip(range(len(a)), appender[i]):
                a[i2].append(i3)
    return a

_cache = range(1, monthrange(_year, _month)[1]+1)
_datedata = DateData(_month)
write_once("_temp.txt", tabulate(list_in_list(_cache, _datedata.length_memos()), headers=["day", "memos"]))
_datedata.save_config()

run("cat _temp.txt | less", shell=True)
remove("_temp.txt")

_term = TerminalMenu(list(map(str, _cache)))
_term.show()

if len(_datedata.data[_term.chosen_menu_index].memos) == 0:
    _exit(0)

_term = TerminalMenu(_datedata.data[_term.chosen_menu_index].names())
_term.show()

run(f"glow {NOTE_FOLDER}/{_term.chosen_menu_entry}", shell=True)
