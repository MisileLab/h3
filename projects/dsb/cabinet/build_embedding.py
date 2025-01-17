from json import dumps
from pathlib import Path
from pickle import loads

from twscrape import Tweet # pyright: ignore[reportMissingTypeStubs]

data: dict[int, list[Tweet]] = {}

for i in Path("results").glob("*.pkl"):
  data[int(i.name.strip(".pkl"))] = loads(i.read_bytes())

for i in Path("results_normal").glob("*.pkl"):
  data[int(i.name.strip(".pkl"))] = loads(i.read_bytes())

Path("embeddings").mkdir(exist_ok=True)

for k, v in data.items():
  with open(Path("embeddings", f"{k}.jsonl"), "w") as f:
    for n, i in enumerate(v):
      _ = f.write(dumps({
          "custom_id": f"request-{n}",
          "method": "POST",
          "url": "/v1/embeddings",
          "body": {
            "model": "text-embedding-3-large",
            "input": i.rawContent
          }
      }, ensure_ascii=False) + "\n")

