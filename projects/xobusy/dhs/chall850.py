from requests import get
from os import mkdir
from os.path import isdir

if not isdir('a'):
 mkdir('a')

url = "http://host3.dreamhack.games:17307"
imgs = ["10", "17", "13", "7","16", "8", "14", "2", "9", "5", "11", "6", "12", "3", "0", "19", "4", "15", "18", "1"]

for i in imgs:
 a = get(f"{url}/static/images/{i}.png")
 print(a.status_code)
 if a.ok:
  with open(f"a/{i}.png", "wb") as f:
   f.write(a.content)

