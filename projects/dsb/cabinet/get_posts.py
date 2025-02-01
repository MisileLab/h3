from time import sleep
from secrets import SystemRandom
from asyncio import run
from pathlib import Path

from loguru import logger
from twscrape import API, Tweet, gather # pyright: ignore[reportMissingTypeStubs]
from twscrape.logger import set_log_level # pyright: ignore[reportMissingTypeStubs]
from pandas import DataFrame, concat # pyright: ignore[reportMissingTypeStubs]

from lib import get_proxy, read_pickle

def get_value[T](v: T | None) -> T:
  if v is None:
    raise TypeError()
  return v

async def get_tweets(api: API, user_id: int) -> list[Tweet]:
  r = SystemRandom().randint(1, 10)
  logger.info(f"sleep {r} sec")
  sleep(r)
  return await gather(api.user_tweets(user_id)) # pyright: ignore[reportUnknownMemberType]

async def get_user(api: API, userid: int) -> int:
  r = SystemRandom().randint(1, 10)
  logger.info(f"sleep {r} sec")
  sleep(r)
  res = await api.user_by_id(userid) # pyright: ignore[reportUnknownMemberType]
  if res is None:
    raise TypeError("user is None")
  return res.id

set_log_level("DEBUG")
if not Path("./results").is_dir():
  Path("./results").mkdir()

async def main():
  api = API(proxy=get_proxy())
  df = read_pickle("data.pkl")
  df_user = read_pickle("user.pkl")

  for i in df_user: # pyright: ignore[reportUnknownVariableType]
    uid: int = i["id"] # pyright: ignore[reportUnknownVariableType]
    logger.debug(uid)
    if not df.loc[df["id"] == uid].empty: # pyright: ignore[reportUnknownMemberType]
      logger.info("skip because exists")
      continue
    tweets = await get_tweets(api, uid) # pyright: ignore[reportUnknownArgumentType]
    if len(tweets) == 0:
      logger.error(f"no tweets on {uid}, skip it")
      continue
    df_user = concat([df_user, DataFrame({
      "id": uid,
      "name": i["name"],
      "url": i["url"],
      "sucidal": i["sucidal"],
      "tweets": tweets
    })])

if __name__ == "__main__":
  run(main())

