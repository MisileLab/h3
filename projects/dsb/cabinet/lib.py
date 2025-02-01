from pandas import DataFrame, read_pickle as _read_pickle, concat # pyright: ignore[reportMissingTypeStubs]
from twscrape import API # pyright: ignore[reportMissingTypeStubs]
from loguru import logger

from os import getenv
from sys import stdout
from pathlib import Path

logger.remove()
_ = logger.add(stdout, level="DEBUG")

def get_proxy():
  proxy_url = getenv("PROXY_URL")
  proxy_user = getenv("PROXY_USERNAME")
  proxy_pass = getenv("PROXY_PASSWORD")

  prx = (
    None if None in [proxy_url, proxy_user, proxy_pass] else
    f"http://{proxy_user}:{proxy_pass}@{proxy_url}"
  )
  logger.debug(prx)
  return prx

async def get_usernames() -> list[str]:
  api = API()
  lst = await api.pool.accounts_info()
  return [a["username"] for a in lst if not (a["active"] or a["logged_in"])]

def read_pickle(file_path: str) -> DataFrame:
  if Path(file_path).exists():
    _df = _read_pickle(file_path)
    df = DataFrame() if not isinstance(_df, DataFrame) else _df
  else:
    df = DataFrame()
  return df

def is_unique(df: DataFrame, key: str, value: object) -> bool:
  try:
    return df.loc[df[key] == value].empty # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
  except KeyError:
    return True

def append(df: DataFrame, data: dict[str, object]) -> DataFrame:
  return concat([df, DataFrame({k:[v] for k,v in data.items()})])
