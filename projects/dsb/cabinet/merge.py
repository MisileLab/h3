from pandas import DataFrame, Series # pyright: ignore[reportMissingTypeStubs]

from lib import read_pickle, append

df = read_pickle("data.pkl")
df_user = DataFrame()

for i in df.loc[df['confirmed']]: # pyright: ignore[reportUnknownVariableType]
  data: Series = i # pyright: ignore[reportUnknownVariableType]
  del data["confirmed"]
  del data["tweets"]
  df_user = append(df_user, data)

df_user.to_pickle("user.pkl")

