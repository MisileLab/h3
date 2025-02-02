from time import sleep
from secrets import SystemRandom
from asyncio import run
from pathlib import Path
from re import compile, sub

from loguru import logger
from twscrape import API, Tweet, gather # pyright: ignore[reportMissingTypeStubs]
from twscrape.logger import set_log_level # pyright: ignore[reportMissingTypeStubs]

from lib import append, get_proxy, is_unique, read_pickle

url_filter = compile(r"(https?:\/\/)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)")

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

  for i in df_user.loc: # pyright: ignore[reportUnknownVariableType]
    uid: int = i["id"] # pyright: ignore[reportUnknownVariableType]
    logger.debug(uid)
    if not is_unique(df, "id", uid): # pyright: ignore[reportUnknownArgumentType]
      logger.info("skip because exists")
      continue
    data: list[str] = []
    nxt_skip = False
    for j in await get_tweets(api, uid): # pyright: ignore[reportUnknownArgumentType]
      if nxt_skip:
        nxt_skip = False
        continue
      if j.retweetedTweet is not None:
        logger.warning("retweeted")
        nxt_skip = True
        continue
      for mention in j.mentionedUsers:
        logger.info(f"delete {mention.username}")
        j.rawContent = j.rawContent.replace(f"@{mention.username}", "").replace(f"@{mention.displayname}", "")
      j.rawContent = sub(url_filter, "", j.rawContent)
      if j.rawContent == "":
        continue
      data.append(j.rawContent)
    if len(data) == 0:
      logger.error(f"no tweets on {uid}, skip it")
      continue
    df = append(df, {
      "id": uid,
      "name": i["name"],
      "url": i["url"],
      "sucidal": i["sucidal"],
      "data": data,
      "confirmed": False
    })

  df.to_pickle("data.pkl")

if __name__ == "__main__":
  run(main())

