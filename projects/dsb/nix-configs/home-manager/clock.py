from json import dumps
from time import sleep
from datetime import datetime, UTC
from sys import stdout, argv
from pyperclipfix import copy

argv = [i.lower() for i in argv]
la = len(argv)
if la < 2:
  print("it is no argument")
  exit(0)
tz = UTC if argv[1] == "utc" else None
if la >= 3:
  t = datetime.now().astimezone(tz)
  if argv[2] == "copyf":
    copy(t.isoformat())
  else:
    copy(int(datetime.now(UTC).replace(tzinfo=UTC).timestamp()))
  stdout.flush()
  exit()

while True:
  t = datetime.now(tz=tz)
  print(dumps({
    "text": t.strftime("%H:%M:%S"),
    "alt": t.strftime("%H:%M:%S"),
    "tooltip": t.strftime("%Y-%m-%dT%H:%M:%S"),
    "class": "",
    "percentage": 0
  }, separators=(',', ':')))
  stdout.flush()
  sleep(1)

