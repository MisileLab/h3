from polars import col, read_avro

from utils import read_cached_avro

processed = read_avro("processed.avro")
df = read_cached_avro("embedding.avro")

for i in df.iter_rows(named=True):
  processed = processed.filter(col("comment_id") != i["comment_id"]) # pyright: ignore[reportAny]

processed.write_avro("processed.avro")

