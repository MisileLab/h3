from os.path import islink
from os import remove, listdir

for i in listdir("."):
    if islink(i):
        remove(i)
