from webbrowser import open
from pathlib import Path
from sys import argv

p = "./a" if len(argv) < 2 else argv[1]
print(p)

for i in Path(p).read_text().split("\n"):
  print(i)
  open(i)

