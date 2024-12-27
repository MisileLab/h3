from csv import DictReader
from os import getenv
from pickle import dump
from time import sleep
from secrets import SystemRandom
from asyncio import run
from pathlib import Path

from loguru import logger
from twscrape import API, Tweet, User, gather # pyright: ignore[reportMissingTypeStubs]
from twscrape.logger import set_log_level # pyright: ignore[reportMissingTypeStubs]

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

  # 1. get username (or id) from url
  # 2. get id and export to json (for backup) and save to list
  if getenv("parseId") == "0":
    processed: list[str] = []
    with open("./userids", "w") as f:
      with open("./result.csv", "r") as r:
        dr = DictReader(r)
        for i in dr:
          _id = ""
          for j in i["url"].removeprefix("https://x.com/"):
            if j in ["/", "?"]:
              break
            _id += j
          logger.debug(_id)
          if _id not in processed:
            user_id = await get_id_by_name(api, _id)
            _ = f.write(f"{user_id}\n")
            processed.append(_id)
    del processed

  # 3. pull posts for user in users
  id = "sample"
  start_point = getenv("num")
  start_point = 0 if start_point is None else int(start_point)
  i = 0

  with open("./userids", "r") as f:
    while True:
      id = f.readline().strip("\n")
      if id == "":
        break
      id = int(id)
      if i > 0:
        i -= 1
        continue
      logger.debug(id)
      tweets = await get_tweets(api, id)
      dump(tweets, open(f"./results/{id}.pkl", "wb"))

if __name__ == "__main__":
  run(main())

