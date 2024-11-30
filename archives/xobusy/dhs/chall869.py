from pathlib import Path
from string import printable
from subprocess import run
a = Path('flag.enc').read_text().split("0X")[1:]
c = ""
d = 0

while True:
  for i in printable:
    Path("flag.txt").write_text(c + i)
    run("./prob")
    b = Path('tmp.enc').read_text().split("0X")[1:]
    if a[d] == b[d]:
      c += i
      d += 1
      print(c)
      break
