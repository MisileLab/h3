from googlesearch import search # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from requests.exceptions import HTTPError, ProxyError
from requests import Timeout
from loguru import logger

from os import getenv
from csv import DictWriter
from http import HTTPStatus
from time import sleep
from secrets import SystemRandom

from lib import get_proxy

proxy = get_proxy()
logger.info(proxy)

base_query = "site:x.com"
suicidal = f"{base_query} #자살"
logger.debug(suicidal)

data_num = 4000
result_interval = 100
sleep_interval_min = 0
sleep_interval_max = 20
start_num = int(getenv("start_num", 0))

def search_res(query: str, start_num: int):
  res = list(search(query, advanced=True, region="kr", num_results=result_interval, start_num=start_num, safe=None, ssl_verify=None, proxy=proxy)) # pyright: ignore[reportArgumentType]
  logger.debug(res)
  return res

with open("suicidal.csv", "w", newline='') as f:
  dw = DictWriter(f, fieldnames=["title", "url", "description"])
  dw.writeheader()
  i = start_num
  while i <= data_num:
    try:
      logger.debug(f"{i}, begin")
      datas = search_res(suicidal, i)
      logger.debug(f"{i}, end")
    except HTTPError as e:
      if e.response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
        logger.warning("retry after 10 mins")
        sleep(60 * 10)
        continue
      raise e
    except (Timeout, ProxyError):
      logger.warning("let's try another interval")
      sleep(10)
      continue
    if datas == []:
      print("data is None")
      break
    for data in datas:
      logger.info(data.title) # pyright: ignore[reportAny]
      dw.writerow({"title": data.title, "url": data.url, "description": data.description}) # pyright: ignore[reportAny]
    i += result_interval
    r = SystemRandom().randint(sleep_interval_min, sleep_interval_max)
    logger.debug(f"sleep {r} secs")
    sleep(r)

