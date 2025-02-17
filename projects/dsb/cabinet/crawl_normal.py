from httpx import ConnectTimeout
from loguru import logger
from pandas import DataFrame # pyright: ignore[reportMissingTypeStubs]
from twscrape import User # pyright: ignore[reportMissingTypeStubs]

from secrets import SystemRandom
from asyncio import run
from time import sleep

from lib import is_unique, read_pickle, append, write_to_pickle, User as dUser, api

sleep_interval_min = 0
sleep_interval_max = 20
min_depth = 1
max_depth = 3
max_following_count = 50
max_user_follower_count = 5000
max_retry = 3

ignore_list: set[int] = set()

class NotEnoughData(Exception):
  def __init__(self, userid: int):
    super().__init__()
    ignore_list.add(userid)

async def search_res(df: DataFrame, userid: int, max_depth: int, depth: int = 0) -> User | None:
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
  while (
    selected_following.verified
    or selected_following.followersCount > max_user_follower_count
    or (depth == max_depth and not is_unique(df, "uid", selected_following.id))
  ):
    if len(followings) == 1:
      logger.warning("user has only non-normal users")
      raise NotEnoughData(selected_following.id)
    logger.info(f"skipping {selected_following.displayname}")
    if selected_following.verified:
      logger.info(f"{selected_following.displayname} is verified")
    elif selected_following.followersCount > max_user_follower_count:
      logger.info(f"{selected_following.displayname} has too many followers ({selected_following.followersCount})")
    else:
      logger.info(f"{selected_following.displayname} is already in the list")
    followings.remove(selected_following)
    selected_following = SystemRandom().choice(followings)
  logger.debug(f"selected: {selected_following.displayname}")
  try:
    res = await search_res(df, selected_following.id, max_depth, depth+1)
  except NotEnoughData:
    logger.error(f"not enough data on first, {depth} {min_depth}")
    if depth-1 > min_depth:
      return await search_res(df, userid, max_depth, depth-1)
    else:
      raise NotEnoughData(selected_following.id)
  if res is None and depth > min_depth and depth < max_depth:
    logger.warning("res's return is None")
    try:
      return await search_res(df, userid, max_depth, depth)
    except NotEnoughData:
      logger.error(f"not enough data on second, {depth} {min_depth}")
      if depth-1 > min_depth:
        return await search_res(df, userid, max_depth, depth-1)
      else:
        raise NotEnoughData(userid)
  return res if res is not None else user

async def subroutine(i: dict[str, object], df: DataFrame, retry: int = 0):
  raw_user = dUser.model_validate(i)
  userid = raw_user.uid
  try:
    user = await search_res(df, userid, max_depth)
  except NotEnoughData:
    logger.error(f"{userid} has not enough data, skipping")
    return
  except ConnectTimeout:
    logger.error(f"{userid} timed out, retry again, current: {retry}")
    if retry >= max_retry:
      logger.error(f"{userid} reached max retry, skipping")
      return
    await subroutine(i, df, retry+1)
    return
  if user is None:
    logger.error(f"{userid} not found, skipping")
    return
  logger.info(f"{user.username}: {user.displayname}")
  if is_unique(df, "uid", user.id):
    df = append(df, dUser(uid=user.id, name=user.username, url=user.url, suicidal=False))
  else:
    logger.error(f"{user.id} is already in the list, this is a bug")
  return

async def main():
  df = read_pickle("user.pkl")
  for i in df.loc[df['suicidal']].to_dict('records'): # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    await subroutine(i, df) # pyright: ignore[reportUnknownArgumentType]
  write_to_pickle(df, "user.pkl")

if __name__ == "__main__":
  run(main())

