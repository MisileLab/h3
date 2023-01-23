from requests import get
from os import chdir, getcwd, listdir
from os.path import splitext, isfile
from shutil import move

chdir('problems')

for i in [i for i in listdir(getcwd()) if isfile(i)]:
    _req = get(f'https://solved.ac/api/v3/problem/lookup?problemIds={splitext(i)[0]}')
    print(_req.json())
    if _req.json()[0]["level"] == 1:
        move(i, f"V/{i}")
