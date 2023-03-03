from calendar import monthrange
from datetime import datetime
from tabulate import tabulate

def list_in_list(*appender: list):
    a = []
    for i in range(len(appender)):
        if i == 0:
            a.extend([i] for i in appender[i])
        else:
            for i2, i3 in zip(range(len(a)), appender[i]):
                a[i2].append(i3)
    return a

print(list_in_list(["a", "b", "c"], [3, 1, 2], [True, False, True]))

today = datetime.now()
_month = list_in_list(list(range(1, monthrange(today.year, today.month)[1]+1)))

print(tabulate(_month, headers="month"))
