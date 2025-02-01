from twscrape import API # pyright: ignore[reportMissingTypeStubs]
from pandas import DataFrame, concat # pyright: ignore[reportMissingTypeStubs]
from loguru import logger

from asyncio import run
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


async def main():
  df = read_pickle("user.pkl")
  api = API(proxy=proxy)

  for suicidal_tag in suicidals:
    logger.info(suicidal_tag)
    async for tweet in api.search(f"#{suicidal_tag}"): # pyright: ignore[reportUnknownMemberType]
      user = tweet.user
      logger.info(user.username)
      if df.loc[df["id"] == user.id].empty: # pyright: ignore[reportUnknownMemberType]
        df = concat([df, DataFrame({
          "id": user.id,
          "name": user.username,
          "url": user.url,
          "suicidal": True
        })])
      r = SystemRandom().randint(0, 10)
      logger.debug(f"sleep {r} secs")
      sleep(r)
  df.to_pickle("user.pkl")

if __name__ == "__main__":
  run(main())
