from lib import read_pickle, write_to_pickle

df = read_pickle("user.pkl")
df = df.drop_duplicates(subset=["uid"])
write_to_pickle(df, "user.pkl")
