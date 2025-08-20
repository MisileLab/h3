from dataclasses import dataclass
from pickle import loads, dumps
from pathlib import Path

from polars import DataFrame, read_parquet

@dataclass
class Data:
  hope: int = 80
  fear: int = 15

data_file = Path("./data.pkl")
if data_file.exists():
  data: Data = loads(data_file.read_bytes()) # pyright: ignore[reportAny]
else:
  data = Data()

parquet_file = Path("./data.parquet")
if parquet_file.exists():
  df: DataFrame = read_parquet(parquet_file)
else:
  df = DataFrame()

def save_data():
  _ = df.write_parquet(parquet_file)
  _ = data_file.write_bytes(dumps(data)) 

def concat(data: DataFrame):
  global df
  df = df.vstack(data)

def add_hope(amount: int):
  """
  Add hope to Scalar.
  returns current hope
  """
  hope_result = min(max(amount + data.hope, 100), 0)
  data.hope = hope_result
  concat(DataFrame({"type": "hope", "amount": hope_result}))
  save_data()
  return f"current hope: {hope_result}"

def add_fear(amount: int):
  """
  Add fear to Scalar.
  returns current fear
  """
  fear_result = min(max(amount + data.fear, 100), 0)
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

