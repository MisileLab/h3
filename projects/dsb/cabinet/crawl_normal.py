from loguru import logger
from twscrape import User, API # pyright: ignore[reportMissingTypeStubs]

from secrets import SystemRandom
from asyncio import run
from time import sleep
from copy import deepcopy

from lib import get_proxy, is_unique, read_pickle, append, write_to_pickle, User as dUser

proxy = get_proxy()
api = API(proxy=proxy)

sleep_interval_min = 0
sleep_interval_max = 20
min_depth = 1
max_depth = 3
max_following_count = 50
max_user_follower_count = 5000
ignore_list: set[int] = set()

class NotEnoughData(Exception):
  def __init__(self, userid: int):
    super().__init__()
    ignore_list.add(userid)

async def search_res(userid: int, max_depth: int, depth: int = 0) -> User | None:
  sleep_sec = SystemRandom().randint(1, 3)
  logger.info(f"sleep {sleep_sec} secs")
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
  for i in followings:
    if i.id in ignore_list:
      followings.remove(i)
  if len(followings) == 0:
    logger.warning("user has no following")
    raise NotEnoughData(userid)
  selected_following = SystemRandom().choice(followings)
  while selected_following.verified or selected_following.followersCount > max_user_follower_count:
    if len(followings) == 1:
      logger.warning("user has only non-normal users")
      raise NotEnoughData(selected_following.id)
    logger.info(f"skipping {selected_following.displayname}")
    if selected_following.verified:
      logger.info(f"{selected_following.displayname} is verified")
    else:
      logger.info(f"{selected_following.displayname} has too many followers ({selected_following.followersCount})")
    followings.remove(selected_following)
    selected_following = SystemRandom().choice(followings)
  logger.debug(f"selected: {selected_following.displayname}")
  try:
    res = await search_res(selected_following.id, max_depth, depth+1)
  except NotEnoughData:
    logger.error(f"not enough data on first, {depth} {min_depth}")
    if depth-1 > min_depth:
      return await search_res(userid, max_depth, depth-1)
    else:
      raise NotEnoughData(selected_following.id)
  if res is None and depth > min_depth and depth < max_depth:
    logger.warning("res's return is None")
    try:
      return await search_res(userid, max_depth, depth)
    except NotEnoughData:
      logger.error(f"not enough data on second, {depth} {min_depth}")
      if depth-1 > min_depth:
        return await search_res(userid, max_depth, depth-1)
      else:
        raise NotEnoughData(userid)
  return res if res is not None else user

async def main():
  df = read_pickle("user.pkl")
  for i in deepcopy(df.loc[df['suicidal']]): # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
    raw_user = dUser.model_validate(i)
    userid = raw_user.uid
    try:
      user = await search_res(userid, max_depth)
    except NotEnoughData:
      logger.error(f"{userid} has not enough data, skipping")
      continue
    if user is None:
      logger.error(f"{userid} not found, skipping")
      continue
    logger.info(f"{user.username}: {user.displayname}")
    if is_unique(df, "uid", user.id):
      df = append(df, dUser(uid=user.id, name=user.username, url=user.url, suicidal=False))
  write_to_pickle(df, "user.pkl")

if __name__ == "__main__":
  run(main())

