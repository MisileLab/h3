from pandas import DataFrame # pyright: ignore[reportMissingTypeStubs]

from lib import read_pickle

filename = input("filename: ")
df = read_pickle(filename)
DataFrame({
  k: v for k, v in zip(df.keys(), df.values)} # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType, reportUnknownVariableType, reportAny]
).to_pickle(filename)

