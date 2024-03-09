from requests import get
from os import chdir, getcwd, listdir, mkdir
from os.path import splitext, isfile, isdir
from shutil import move
from pathlib import Path

chdir('problems')

def move_path(frompath: str, dest: str):
  parent = Path(dest).parent.absolute()
  if isdir(parent) == False:
    mkdir(parent)
  move(frompath, dest)

for i in [i for i in listdir(getcwd()) if isfile(i)]:
  ver = get(f'https://solved.ac/api/v3/problem/lookup?problemIds={splitext(i)[0]}').json()[0]["level"]
  print(ver)
  if ver == 1:
    move_path(i, f"BronzeV/{i}")
  elif ver == 2:
    move_path(i, f"BronzeIV/{i}")
  elif ver == 3:
    move_path(i, f"BronzeIII/{i}")
  elif ver == 4:
    move_path(i, f"BronzeII/{i}")
  elif ver == 8:
    move_path(i, f"SilverIII/{i}")
