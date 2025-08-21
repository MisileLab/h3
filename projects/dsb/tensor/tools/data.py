from dataclasses import dataclass
from pickle import dumps
from pathlib import Path

from polars import DataFrame

from ..libraries.utils import read_parquet, concat as utils_concat, read_pickle

@dataclass
class Data:
  hope: int = 80
  fear: int = 15

data_file = Path("./data.pkl")
df_file = Path("./data.parquet")
data = read_pickle(data_file, Data)
df = read_parquet(df_file)

def save_data():
  _ = df.write_parquet(df_file)
  _ = data_file.write_bytes(dumps(data)) 

def concat(data: DataFrame):
  global df
  df = utils_concat(df, data)

def add_hope(amount: int):
  """
  Add hope to Scalar.
  returns current hope
  """
  print("Adding hope:", amount)
  hope_result = min(max(amount + data.hope, 0), 100)
  data.hope = hope_result
  concat(DataFrame({"type": "hope", "amount": hope_result}))
  save_data()
  return f"current hope: {hope_result}"

def add_fear(amount: int):
  """
  Add fear to Scalar.
  returns current fear
  """
  print("Adding fear:", amount)
  fear_result = min(max(amount + data.fear, 0), 100)
  data.fear = fear_result
  concat(DataFrame({"type": "fear", "amount": fear_result}))
  save_data()
  return f"current fear: {fear_result}"

def get_current_status():
  """
  Get current status of Scalar.
  returns current hope and fear
  """
  return f"current hope: {data.hope}, current fear: {data.fear}"

functions = [add_hope, add_fear]

