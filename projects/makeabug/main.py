# Performance so use numpy
from numpy import array

bugs = array([
    "Input integer max value: 2147483647",
    "Input negative integer value in require not negative number (ex: pay) : -1",
    "Input integer min value: -2147483648",
    "Using minecraft bugs",
    "Using condition with minecraft system (ex: when jump, remove item - so if you want to jump, drop item and jump and get it)",
    "Using lag"
])

for i in bugs:
    print(i)
    input("Did you test it?")
