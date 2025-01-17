from csv import DictReader, DictWriter
from pathlib import Path

writed: list[str] = []

with open("normal.csv", "r", newline="") as f:
  dr = DictReader(f)
  with open("result_normal.csv", "w", newline="") as res:
    dw = DictWriter(res, fieldnames=Path("normal.csv").read_text().split("\n")[0].split(","))
    dw.writeheader()
    for i in dr:
      if i["url"] in writed:
        print("skip")
        continue
      dw.writerow(i)
      writed.append(i["url"])
