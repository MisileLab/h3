from contextlib import suppress
from pandas import DataFrame, Series, read_pickle as _read_pickle, concat # pyright: ignore[reportMissingTypeStubs]
from twscrape import API # pyright: ignore[reportMissingTypeStubs]
from loguru import logger
from pydantic import BaseModel, Field

from os import getenv
from sys import stdout
from pathlib import Path

logger.remove()
_ = logger.add(stdout, level="DEBUG")

class User(BaseModel):
  uid: int = Field(..., strict=False)
  name: str
  suicidal: bool
  url: str

class Data(User):
  data: list[str]
  confirmed: bool = False

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

def write_to_pickle(df: DataFrame, file_path: str) -> None:
  with suppress(ValueError):
    df = df.reset_index()
  with suppress(KeyError):
    del df["level_0"]
  with suppress(KeyError):
    del df["index"]
  df.to_pickle(file_path)

def is_unique(df: DataFrame, key: str, value: object) -> bool:
  try:
    return df.loc[df[key] == value].empty # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
  except KeyError:
    return True

def append(df: DataFrame, data: dict[str, object] | Series | BaseModel) -> DataFrame:
  if isinstance(data, BaseModel):
    data = Series(data.model_dump())
  elif isinstance(data, dict):
    data = Series(data)
  return concat([df, data.to_frame().T], ignore_index=True)
