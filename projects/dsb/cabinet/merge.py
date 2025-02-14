from pandas import DataFrame # pyright: ignore[reportMissingTypeStubs]

from lib import read_pickle, append, write_to_pickle, User, Data

df = read_pickle("data.pkl")
df_user = DataFrame()

for _i in df.loc[df['confirmed']].to_dict('records'): # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
  data = Data.model_validate(_i)
  df_user = append(df_user, User.model_validate(data))

write_to_pickle(df_user, "user.pkl")

