from lib import read_pickle

df = read_pickle("user.pkl")
df = df.drop_duplicates(subset=["id"])
df.to_pickle("user.pkl")
