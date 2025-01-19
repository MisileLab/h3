from loguru import logger
from twscrape import User, API # pyright: ignore[reportMissingTypeStubs]

from os import listdir
from csv import DictWriter
from secrets import SystemRandom
from asyncio import run
from pathlib import Path
from sys import stdout
from time import sleep

from lib import get_proxy

logger.remove()
_ = logger.add(stdout, level="DEBUG")
proxy = get_proxy()
logger.info(proxy)
api = API(proxy=proxy)

class NotEnoughData(Exception):
  pass

sleep_interval_min = 0
sleep_interval_max = 20
min_depth = 2
max_depth = 4
max_following_count = 50
max_user_follower_count = 5000

async def search_res(userid: int, max_depth: int, depth: int = 0) -> User | None:
  sleep_sec = SystemRandom().randint(1, 5)
  logger.info("sleep {sleep_sec} secs")
  sleep(sleep_sec)
  if depth > max_depth:
    return None
  logger.debug(f"searching {userid}, depth: {depth}")
  user = await api.user_by_id(userid) # pyright: ignore[reportUnknownMemberType]
  if user is None:
    return None
  followings: list[User] = []
  async for i in api.following(userid, limit=max_following_count): # pyright: ignore[reportUnknownMemberType]
    followings.append(i)
  if len(followings) == 0:
    logger.warning("user has no following")
    raise NotEnoughData()
  selected_following = SystemRandom().choice(followings)
  while selected_following.verified or selected_following.followersCount > max_user_follower_count:
    if len(followings) == 1:
      logger.warning("user has only non-normal users")
      raise NotEnoughData()
    logger.info(f"skipping {selected_following.displayname}")
    if selected_following.verified:
      logger.info(f"{selected_following.displayname} is verified")
    else:
      logger.info(f"{selected_following.displayname} has too many followers ({selected_following.followersCount})")
    followings.remove(selected_following)
    selected_following = SystemRandom().choice(followings)
  logger.debug(f"selected: {selected_following.displayname}")
  res = await search_res(selected_following.id, max_depth, depth+1)
  if res is None and depth < min_depth:
    try:
      return await search_res(userid, max_depth, depth)
    except NotEnoughData:
      return None
  return res if res is not None else user

exist = Path("normal.csv").is_file()

async def main():
  with open("normal.csv", "a" if exist else "w", newline='') as f:
    dw = DictWriter(f, fieldnames=["id", "name", "url"])
    if not exist:
      dw.writeheader()
    for i in listdir("results"):
      try:
        user = await search_res(int(i.removesuffix(".pkl")), max_depth)
      except NotEnoughData:
        logger.error(f"user {i.removesuffix('.pkl')} has not enough data, skipping")
        continue
      if user is None:
        logger.error(f"user {i.removesuffix('.pkl')} not found, skipping")
        continue
      logger.info(f"{user.username}: {user.displayname}")
      dw.writerow({"id": user.id, "name": user.username, "url": user.url})

if __name__ == "__main__":
  run(main())

