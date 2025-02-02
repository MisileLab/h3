from pydoc import pager

from loguru import logger
from pandas import DataFrame, Series # pyright: ignore[reportMissingTypeStubs]

from lib import read_pickle, append

data = read_pickle("data.pkl")
data_res = DataFrame()

for i in data.loc: # pyright: ignore[reportUnknownVariableType]
  i: Series
  if i["confirmed"] is True:
    logger.info("skip because already confirmed")
    data_res = append(data_res, i) # pyright: ignore[reportUnknownArgumentType]
    continue
  tweets: list[str] = i["tweets"] # pyright: ignore[reportAssignmentType]
  pager("\n--sep--".join(i for i in tweets if i.count("자살")+i.count("자해") != 0))
  full = input("do you want to see full? [y/n]").lower() == "y"
  if full:
    pager("\n--sep--".join(tweets))
  suicidal = input("is this suicidal? [y/n]")
  while suicidal.lower() not in ["y", "n"]:
    suicidal = input("is this suicidal? [y/n]")
  i["suicidal"] = suicidal.lower() == "y"
  i["confirmed"] = True
  data_res = append(data_res, i) # pyright: ignore[reportUnknownArgumentType]

data_res.to_pickle("data.pkl")

