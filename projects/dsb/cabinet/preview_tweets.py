from copy import deepcopy

from pandas import DataFrame # pyright: ignore[reportMissingTypeStubs]
from pypager.pager import Pager # pyright: ignore[reportMissingTypeStubs]
from pypager.source import StringSource # pyright: ignore[reportMissingTypeStubs]

from lib import Data, read_pickle, write_to_pickle

data = read_pickle("data.pkl")
data_res = deepcopy(data)
try:
  for _i in data.loc[data["confirmed"] == False].to_dict('records'): # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType] # noqa: E712
    i = Data.model_validate(_i)
    tweets: list[str] = i.data
    suicidal_comments: str = "\n--sep--\n".join(i for i in tweets if i.count("자살")+i.count("자해") != 0)
    if suicidal_comments != "":
      p = Pager()
      _ = p.add_source(StringSource(suicidal_comments))
      p.run()
    elif i.suicidal:
      print("previously suicidal but none found")
    else:
      print("suicidal none found")
    full = suicidal_comments == "" or input("do you want to see full? [y/n]: ").lower() == "y"
    if full:
      p = Pager()
      _ = p.add_source(StringSource("\n--sep--\n".join(i for i in tweets)))
      p.run()
    suicidal = input("is this suicidal? (if not normal message and it is something like news, input 'r') [y/n/r]: ")
    while suicidal.lower() not in ["y", "n", "r"]:
      suicidal = input("is this suicidal? (if not normal message and it is something like news, input 'r') [y/n/r]: ")
    if suicidal.lower() == "r":
      print("remove")
      _d_res = data_res[data_res["uid"] != i.uid] # pyright: ignore[reportUnknownVariableType]
      if not isinstance(_d_res, DataFrame):
        raise Exception("wtf")
      data_res = _d_res
      continue
    data_res.loc[data_res["uid"] == i.uid, "suicidal"] = suicidal.lower() == "y"
    data_res.loc[data_res["uid"] == i.uid, "confirmed"] = True
except KeyboardInterrupt:
  write_to_pickle(data_res, "data.pkl")
