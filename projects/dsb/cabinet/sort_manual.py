from shutil import move
from pathlib import Path
from re import sub, compile

regex = compile(r"_[1-9]\.jsonl")

while True:
  userid = sub(regex, "", input("Enter name:").removeprefix("normal"))
  print(userid)
  if not Path("./results_normal", f"{userid}.pkl").is_file():
    print("skip")
    continue
  move(f"./results_normal/{userid}.pkl", f"./results/{userid}.pkl")

