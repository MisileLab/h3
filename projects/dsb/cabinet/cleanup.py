from pathlib import Path
from os import getenv

from openai import OpenAI

o = OpenAI(api_key=getenv("OPENAI_KEY"))

for i in Path("./embedding_results").glob("*.json"):
  bid = i.stem
  batch = o.batches.retrieve(bid)
  if batch.status == "completed":
    print(f"{batch.id} completed")
    output_id = batch.output_file_id
    if output_id is None:
      print(f"{batch.id} has no output file")
      continue
    _ = o.files.delete(output_id)

