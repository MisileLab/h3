from googlesearch import SearchResult, search # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from pandas import concat # pyright: ignore[reportMissingTypeStubs]
from requests import Timeout
from requests.exceptions import HTTPError, ProxyError
from loguru import logger

from http import HTTPStatus
from time import sleep
from secrets import SystemRandom

from lib import get_proxy, read_pickle

proxy = get_proxy()

base_query = "site:x.com"
suicidals = [
  "자해", "자해러", "자해계",
  "자살", "자살시도", "자살충동", "자살사고",
  "죽고싶다", "죽고싶어"
]

data_num = 4000
result_interval = 100
sleep_interval_min = 0
sleep_interval_max = 20

def search_res(query: str, start_num: int):
  res: list[SearchResult] = []
  for i in search( # pyright: ignore[reportUnknownVariableType]
    query,
    advanced=True,
    region="kr",
    num_results=result_interval,
    start_num=start_num,
    safe=None, # pyright: ignore[reportArgumentType]
    ssl_verify=None,
    proxy=proxy
  ):
    if not isinstance(i, SearchResult):
      logger.debug("skip")
      continue
    res.append(i)
  logger.debug(res)
  return res

df = read_pickle("user_raw.pkl")

for suicidal_tag in suicidals:
  logger.info(suicidal_tag)
  i = 0
  while i <= data_num:
    try:
      logger.debug(f"{i}, begin")
      datas = search_res(base_query + f" #{suicidal_tag}", i)
      logger.debug(f"{i}, end")
    except HTTPError as e:
      if e.response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
        logger.warning("retry after 10 mins")
        sleep(60 * 10)
        continue
      raise e
    except (Timeout, ProxyError, ConnectionError):
      logger.warning("let's try another interval")
      sleep(10)
      continue
    if datas == []:
      print("data is None")
      break
    for data in datas:
      logger.info(data.title) # pyright: ignore[reportUnknownArgumentType]
      if df.loc[df["url"] == data.url].empty: # pyright: ignore[reportUnknownMemberType]
        df = concat({"title": data.title, "url": data.url, "description": data.description})
    i += result_interval
    r = SystemRandom().randint(sleep_interval_min, sleep_interval_max)
    logger.debug(f"sleep {r} secs")
    sleep(r)

df.to_pickle("user_raw.pkl")
