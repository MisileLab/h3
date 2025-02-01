from pandas import DataFrame, concat # pyright: ignore[reportMissingTypeStubs]
from twscrape import API # pyright: ignore[reportMissingTypeStubs]

from asyncio import run

from lib import read_pickle, get_proxy

async def main():
  api = API(proxy=get_proxy())

  df = read_pickle("user_raw.pkl")
  df_user = read_pickle("user.pkl")
  for i in df.loc[df["id"].isnull()]: # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    url: str = i["url"] # pyright: ignore[reportUnknownVariableType]
    username_val: str = url.removeprefix("https://").removeprefix("http://") # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    username: str = username_val.split("/")[1] # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    print(username) # pyright: ignore[reportUnknownArgumentType]
    user = await api.user_by_login(username) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    if user is None:
      print("skip")
      continue
    df_user = concat([df_user, DataFrame({
      "id": user.id,
      "name": user.username,
      "url": user.url,
      "suicidal": True
    })])
  df_user.to_pickle("user.pkl")

if __name__ == "__main__":
  run(main())
