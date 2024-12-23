from googlesearch import search # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from requests.exceptions import HTTPError, ProxyError
from requests import Timeout

from os import getenv
from csv import DictWriter
from http import HTTPStatus
from time import sleep
from secrets import SystemRandom

proxy_url = getenv("PROXY_URL")
proxy_user = getenv("PROXY_USERNAME")
proxy_pass = getenv("PROXY_PASSWORD")

if None in [proxy_url, proxy_user, proxy_pass]:
  proxy = None
else:
  proxy = f"http://{proxy_user}:{proxy_pass}@{proxy_url}"
print(proxy)

base_query = "site:x.com"
suicidal = f"{base_query} #자살"

data_num = 4000
result_interval = 10
sleep_interval_min = 0
sleep_interval_max = 60
_start = getenv("start_num")
start_num = 1 if _start is None else int(_start)

def search_res(query: str, start_num: int):
  return list(search(query, advanced=True, unique=True, num_results=result_interval, start_num=start_num, lang="ko", safe=None, ssl_verify=None, proxy=proxy)) # pyright: ignore[reportArgumentType]

with open("suicidal.csv", "w", newline='') as f:
  dw = DictWriter(f, fieldnames=["title", "url", "description"])
  dw.writeheader()
  i = start_num
  while i <= data_num:
    try:
      print(i, "begin")
      datas = search_res(suicidal, i)
      print(i, "end")
    except HTTPError as e:
      if e.response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
        print("retry after 10 mins")
        sleep(60 * 10)
        continue
      raise e
    except (Timeout, ProxyError):
      print("let's try another interval")
      sleep(10)
      continue
    for data in datas:
      print(data.title) # pyright: ignore[reportAny]
      dw.writerow({"title": data.title, "url": data.url, "description": data.description}) # pyright: ignore[reportAny]
    i += result_interval
    r = SystemRandom().randint(sleep_interval_min, sleep_interval_max)
    print(f"rest for {r}")
    sleep(r)

