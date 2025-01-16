from loguru import logger
from twscrape import User, API # pyright: ignore[reportMissingTypeStubs]

from os import listdir
from csv import DictWriter
from secrets import SystemRandom
from asyncio import run

from lib import get_proxy

proxy = get_proxy()
logger.info(proxy)
api = API(proxy=proxy)

sleep_interval_min = 0
sleep_interval_max = 20
min_depth = 2
max_depth = 4

async def search_res(userid: int, max_depth: int, depth: int = 0) -> User | None:
  if depth > max_depth:
    return None
  user = await api.user_by_id(userid) # pyright: ignore[reportUnknownMemberType]
  if user is None:
    return None
  index = 0
  if user.followersCount == 1:
    following_index = 0
  elif user.followersCount > 1:
    following_index = SystemRandom().randint(0, user.followersCount-1)
  else:
    return None
  selected_following: User | None = None
  logger.debug(f"getting {userid}'s following, index: {following_index}")
  async for i in api.following(user.id): # pyright: ignore[reportUnknownMemberType]
    if index > following_index:
      break
    selected_following = i
    index += 1
    logger.debug(index)
  if selected_following is None:
    raise ValueError("no following?")
  res = await search_res(selected_following.id, max_depth, depth+1)
  if res is None and depth < min_depth:
    return await search_res(userid, max_depth, depth)
  return res if res is not None else user

async def main():
  with open("normal.csv", "w", newline='') as f:
    dw = DictWriter(f, fieldnames=["id", "name", "url"])
    dw.writeheader()
    for i in listdir("results"):
      user = await search_res(int(i.removesuffix(".pkl")), max_depth)
      if user is None:
        logger.error(f"user {i.removesuffix('.pkl')} not found, skipping")
        continue
      logger.info(user.id_str)
      dw.writerow({"id": user.id, "name": user.id_str, "url": user.url})

if __name__ == "__main__":
  run(main())

