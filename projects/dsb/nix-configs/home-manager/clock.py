from json import dumps
from time import sleep
from datetime import datetime, UTC
from sys import stdout, argv

while True:
  try:
    t = datetime.now(tz=UTC if argv[1] == "UTC" else None)
  except IndexError:
    t = datetime.now()
  print(dumps({
    "text": t.strftime("%H:%M:%S"),
    "alt": t.strftime("%H:%M:%S"),
    "tooltip": t.strftime("%Y-%m-%dT%H:%M:%S"),
    "class": "",
    "percentage": 0
  }, separators=(',', ':')))
  stdout.flush()
  sleep(1)

