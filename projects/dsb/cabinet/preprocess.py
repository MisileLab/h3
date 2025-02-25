from pickle import dumps
from pathlib import Path

from tqdm.auto import tqdm
from polars import read_ndjson, col, Int64

data: dict[str, list[list[list[float]]]] = {
  "suicidal": [],
  "normal": []
}

for i in tqdm(list(Path("embedding_results").glob("*.jsonl"))):
  df = read_ndjson(i)
  df = df.with_columns(
    col("custom_id").str.replace("request-", "").cast(Int64).alias("sort_key")
  ).sort("sort_key").drop("sort_key")
  embedding: list[list[float]] = [x["body"]["data"][0]["embedding"] for x in df["response"]] # pyright: ignore[reportAny]
  data["normal" if not i.name.startswith("suicidal") else "suicidal"].append(embedding)

_ = Path("embedding.pkl").write_bytes(dumps(data))