from requests import get
from os import chdir, getcwd, listdir
from os.path import splitext, isfile
from shutil import move

chdir('problems')

for i in [i for i in listdir(getcwd()) if isfile(i)]:
    ver = get(f'https://solved.ac/api/v3/problem/lookup?problemIds={splitext(i)[0]}').json()[0]["level"]
    if ver == 1:
        move(i, f"BronzeV/{i}")
    elif ver == 2:
        move(i, f"BronzeIV/{i}")
