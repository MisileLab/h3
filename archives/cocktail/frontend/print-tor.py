from os.path import isfile
from time import sleep
from pathlib import Path

while not isfile("/var/lib/tor/cocktail/hostname"):
 sleep(1)

print(Path("/var/lib/tor/cocktail/hostname").read_text())

