from pydoc import pager

from pandas import DataFrame # pyright: ignore[reportMissingTypeStubs]

from lib import Data, read_pickle, append, write_to_pickle

data = read_pickle("data.pkl")
data_res = DataFrame()

for _i in data.loc[data["confirmed"] is False]: # pyright: ignore[reportUnknownVariableType]
  i = Data.model_validate(_i)
  tweets: list[str] = i.data
  pager("\n--sep--".join(i for i in tweets if i.count("자살")+i.count("자해") != 0))
  full = input("do you want to see full? [y/n]").lower() == "y"
  if full:
    pager("\n--sep--".join(tweets))
  suicidal = input("is this suicidal? [y/n]")
  while suicidal.lower() not in ["y", "n"]:
    suicidal = input("is this suicidal? [y/n]")
  i.suicidal = suicidal.lower() == "y"
  i.confirmed = True
  data_res = append(data_res, i)
  write_to_pickle(data_res, "data.pkl")
