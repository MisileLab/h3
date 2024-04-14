from httpx import Client

from subprocess import run

TOR_PROXY="socks5://127.0.0.1:9050"

print("let's check tor's proxy (note: you must use ./torrc for tor configure page)")
with Client(proxy=TOR_PROXY) as c:
 r = c.get("http://3i2es3fbaoergypokczbtmejrpg4utikp7hzyxoh5rm7ejmizaqos7id.onion")
 print(r.status_code)
 if r.is_client_error:
  print("I think tor proxy not configured")
  exit()
 print("Tor configured, let's send to server")
 r = c.post("")

