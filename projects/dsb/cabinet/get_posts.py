from loguru import logger
from pytwitter import Api  # pyright: ignore[reportMissingTypeStubs]
from pytwitter.models import Tweet # pyright: ignore[reportMissingTypeStubs]
from pytwitter.error import PyTwitterError # pyright: ignore[reportMissingTypeStubs]

from csv import DictReader
from os import getenv
from pickle import dump
from time import sleep

# 1. get username (or id) from url
# 2. get id and export to json (for backup) and save to list
if getenv("parseId") == "0":
  current_ids: list[str] = []
  with open("./userids", "w") as f:
    with open("./result.csv", "r") as r:
      dr = DictReader(r)
      for i in dr:
        current_id = ""
        for j in i["url"].removeprefix("https://x.com/"):
          if j in ["/", "?"]:
            break
          current_id += j
        logger.debug(current_id)
        if current_id not in current_ids:
          _ = f.write(f"{current_id}\n")
          current_ids.append(current_id)
  del current_ids

# 3. pull posts for user in users
id = "sample"
start_point = getenv("num")
start_point = 0 if start_point is None else int(start_point)

logger.debug(getenv("BEARER_TOKEN"))
api = Api(getenv("BEARER_TOKEN"))
i = 0
since_id = None

def get_value[T](v: T | None) -> T:
  if v is None:
    raise TypeError()
  return v

def get_tweets(user_id: str, since_id: str | None = None, error: PyTwitterError | None = None) -> list[Tweet]:
  try:
    return api.get_timelines(user_id, max_results=100, since_id=since_id, media_fields=["url", "type", "media_key"], tweet_fields=["attachments", "text", "id"]).data # pyright: ignore[reportAttributeAccessIssue, reportReturnType, reportUnknownMemberType, reportUnknownVariableType]
  except PyTwitterError as e:
    if error is None:
      logger.warning("sleep 10 secs")
      sleep(10)
      return get_tweets(user_id, since_id, e)
    raise e from error

with open("./userids", "r") as f:
  while id != "":
    id = f.readline().strip("\n")
    if i > 0:
      i -= 1
      continue
    since_id = None
    logger.debug(id)
    tweets = get_tweets(id)
    while len(tweets) == 100:
      logger.debug(len(tweets))
      tweets.extend(get_tweets(get_value(tweets[-1].id)))
    dump(tweets, open(f"./results/{id}.csv", "wb"))

