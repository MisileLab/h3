from os import getenv
from pathlib import Path

from openai import OpenAI
from openai.types import Batch
from pickle import dumps

o = OpenAI(api_key=getenv("OPENAI_KEY"))
batches: list[Batch] = []

for i in Path("embeddings").glob("*"):
  print(i.name)
  file = o.files.create(file=open(i, "rb"), purpose="batch")
  batches.append(
    o.batches.create(
      input_file_id=file.id,
      endpoint="/v1/embeddings",
      completion_window="24h",
      metadata={"description": i.name}
    )
  )

_ = Path("batches.pkl").write_bytes(dumps(batches))

