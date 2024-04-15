from httpx import Client

from subprocess import run
from threading import Thread
from asyncio import run

TOR_PROXY="socks5://127.0.0.1:9050"
ONION_ADDRESS=""
VERSION="Midori Sour"

print("let's check tor's proxy (note: you need to do 5000 -> tor onion and add it to ONION_ADDRESS)")
with Client(proxy=TOR_PROXY) as c:
 r = c.get("http://3i2es3fbaoergypokczbtmejrpg4utikp7hzyxoh5rm7ejmizaqos7id.onion")
 print(r.status_code)
 if r.is_client_error:
  print("I think tor proxy not configured")
  exit()
 print("Tor configured, let's send to server")
 print(f"Welcome to cocktail {VERSION}")
 print(f"lets try to connect your {ONION_ADDRESS}")
 r = c.get(ONION_ADDRESS)
 r.raise_for_status()
 if r.text != VERSION:
  print("version is different")
  exit(1)
 print("ok, we connected, now talk :sunglasses:")
 
 

