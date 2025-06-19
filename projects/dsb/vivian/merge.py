from polars import read_avro, DataFrame, concat

from pathlib import Path

df_new = read_avro("new_embedding.avro")
df = read_avro("embedding.avro")

def append(df: DataFrame, data: DataFrame) -> DataFrame:
  return concat([df, data], how="vertical", rechunk=True)

for i in df.iter_rows(named=True):
  df = append(df, df_new)

df.write_avro("embedding.avro")

Path("./new_embedding.avro").unlink()

