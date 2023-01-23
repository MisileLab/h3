from requests import get
from os import chdir, getcwd, listdir
from os.path import splitext, isfile
from shutil import move

chdir('problems')

flist = []
for i in listdir(getcwd()):
    if isfile(i):
        flist.append(i)

print(get(f'https://solved.ac/api/v3/problem/lookup?problemIds={splitext(i)[0]}').json()[0])

for i in flist:
    if get(f'https://solved.ac/api/v3/problem/lookup?problemIds={splitext(i)[0]}').json()[0]["level"] == 1:
        move(i, f"V/{i}")
