from pandas import DataFrame, Series, concat # pyright: ignore[reportMissingTypeStubs]

from lib import read_pickle

df = read_pickle("data.pkl")
df_user = DataFrame()

for i in df.loc[df['confirmed']]: # pyright: ignore[reportUnknownVariableType]
  data: Series = i # pyright: ignore[reportUnknownVariableType]
  del data["confirmed"]
  del data["tweets"]
  df_user = concat([df_user, DataFrame(data)])

df_user.to_pickle("user.pkl")

