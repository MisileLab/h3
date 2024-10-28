from httpx import stream
from tqdm import tqdm
from fake_useragent import UserAgent

from sys import argv
argc = len(argv)

if argc <= 1:
  print("plz insert url")
  exit(1)
elif argc <= 2:
  print("plz insert output file")
  exit(1)

url = argv[1]
output = argv[2]
print(url, output)

ua = UserAgent()
ra = ua.random
print(f"agent: {ra}")

with stream("GET", url, headers={"User-Agent": ra}, follow_redirects=True) as r:
  r.raise_for_status()
  print(r.headers.get("content-length"))
  with tqdm(total=int(r.headers.get("content-length", 0)), unit="B", unit_scale=True) as progress:
    with open(output, "wb") as f:
      for data in r.iter_raw():
        f.write(data)
        progress.update(len(data))

