from json import dumps
from time import sleep
from datetime import datetime, UTC, timedelta
from sys import stdout

while True:
  t = datetime.now(tz=UTC) + timedelta(seconds=1)
  print(dumps({
    "text": t.strftime("%H:%M:%S"),
    "alt": t.strftime("%H:%M:%S"),
    "tooltip": t.strftime("%Y-%m-%dT%H:%M:%S"),
    "class": "",
    "percentage": 0
  }, separators=(',', ':')))
  stdout.flush()
  sleep(1)

