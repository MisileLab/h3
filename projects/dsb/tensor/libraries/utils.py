from pathlib import Path
from pickle import loads
from typing import Callable

from polars import DataFrame, read_parquet as polars_read_parquet

def read_pickle[T](file_path: Path, default: Callable[..., T]) -> T:
  if file_path.exists():
    return loads(file_path.read_bytes()) # pyright: ignore[reportAny]
  return default()

def read_parquet(file_path: Path) -> DataFrame:
  if file_path.exists():
    return polars_read_parquet(file_path)
  return DataFrame()

def concat(df: DataFrame, data: DataFrame) -> DataFrame:
  return df.vstack(data)
