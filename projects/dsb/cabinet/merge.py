from pandas import DataFrame # pyright: ignore[reportMissingTypeStubs]

from lib import read_pickle, write_to_pickle

df = read_pickle("data.pkl")
df_user: DataFrame = df.loc[df['confirmed']] # pyright: ignore[reportUnknownVariableType]
del df_user['data']

write_to_pickle(df_user, "user.pkl") # pyright: ignore[reportUnknownArgumentType]

