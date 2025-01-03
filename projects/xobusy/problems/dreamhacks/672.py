from subprocess import run, PIPE
from re import sub

def strip_ansi_codes(s: str) -> str:
  return sub(r'\x1b\[([0-9,A-Z]{1,2}(;[0-9]{1,2})?(;[0-9]{3})?)?[m|K]?', '', s)

a = run(["objdump", "-d", "./collect_me"], stdout=PIPE)

flag = ""
recording = False
start_function = False
n = 0

for i in a.stdout.decode().split("\n"):
  if i.endswith("<func_659>:"):
    recording = True
  if i.endswith("<func_691>:"):
    print()
    exit()
  if recording:
    if n == 4:
      print(bytes.fromhex(i.split(" ")[-1].split(",")[0][3:]).decode(), end='')
      n = -4
      start_function = False
      continue
    n += 1
