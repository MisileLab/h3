from csv import DictReader
from time import sleep
from secrets import SystemRandom
from asyncio import run
from pathlib import Path
from pickle import dump

from loguru import logger
from twscrape import API, Tweet, User, gather # pyright: ignore[reportMissingTypeStubs]
from twscrape.logger import set_log_level # pyright: ignore[reportMissingTypeStubs]

from lib import get_proxy

def get_value[T](v: T | None) -> T:
  if v is None:
    raise TypeError()
  return v

async def get_tweets(api: API, user_id: int) -> list[Tweet]:
  r = SystemRandom().randint(1, 10)
  logger.info(f"sleep {r} sec")
  sleep(r)
  return await gather(api.user_tweets(user_id)) # pyright: ignore[reportUnknownMemberType]

async def get_id_by_name(api: API, username: str) -> int:
  r = SystemRandom().randint(1, 10)
  logger.info(f"sleep {r} sec")
  sleep(r)
  res: User | None = await api.user_by_login(username) # pyright: ignore[reportUnknownMemberType]
  if res is None:
    raise TypeError("user is None")
  return res.id

set_log_level("DEBUG")
if not Path("./results").is_dir():
  Path("./results").mkdir()

async def main():
  api = API()
  api.proxy = get_proxy()

  with open("./result_normal.csv", "r") as f:
    reader = DictReader(f)
    for i in reader:
      uid = i["id"]
      if uid == "":
        break
      uid = int(uid)
      if Path("results_normal", f"{uid}.pkl").is_file():
        logger.warning(f"skipping {uid}")
        continue
      logger.debug(uid)
      tweets = await get_tweets(api, uid)
      if len(tweets) == 0:
        logger.error(f"no tweets on {uid}, skip it")
        continue
      dump(tweets, open(f"./results_normal/{uid}.pkl", "wb"))

if __name__ == "__main__":
  run(main())

