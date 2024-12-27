from csv import DictReader
from os import getenv
from time import sleep
from secrets import SystemRandom
from asyncio import run
from pathlib import Path
from json import dumps

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
  start_uid = int(getenv("start_uid", -1))

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
  uid = "sample"
  i = 0

  with open("./userids", "r") as f:
    while True:
      uid = f.readline().strip("\n")
      if uid == "":
        break
      uid = int(uid)
      if start_uid not in [-1, uid]:
        logger.debug("skipping")
        continue
      start_uid = -1
      if i > 0:
        i -= 1
        continue
      logger.debug(uid)
      tweets = await get_tweets(api, uid)
      Path(f"./results/{uid}.json").touch()
      _ = Path(f"./results/{uid}.json").write_text(dumps(
        [t.dict() for t in tweets]
      ))

if __name__ == "__main__":
  run(main())

