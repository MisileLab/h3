from json import dumps
from pathlib import Path
from pickle import loads
from dataclasses import dataclass

from twscrape import Tweet # pyright: ignore[reportMissingTypeStubs]

@dataclass
class Data:
  t: list[Tweet]
  suicidal: bool

data: dict[int, Data] = {}

for i in Path("results").glob("*.pkl"):
  data[int(i.name.strip(".pkl"))] = Data(loads(i.read_bytes()), True) # pyright: ignore[reportAny]

for i in Path("results_normal").glob("*.pkl"):
  data[int(i.name.strip(".pkl"))] = Data(loads(i.read_bytes()), False) # pyright: ignore[reportAny]

Path("embeddings").mkdir(exist_ok=True)

for k, v in data.items():
  # TODO: check token limit when build embedding and if exceeds, split jsonl
  with open(Path("embeddings", f"{'suicidal' if v.suicidal else 'normal'}{k}.jsonl"), "w") as f:
    for n, i in enumerate(v.t):
      _ = f.write(dumps({
          "custom_id": f"request-{n}",
          "method": "POST",
          "url": "/v1/embeddings",
          "body": {
            "model": "text-embedding-3-large",
            "input": i.rawContent
          }
      }, ensure_ascii=False) + "\n")

