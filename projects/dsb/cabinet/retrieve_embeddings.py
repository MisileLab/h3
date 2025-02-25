from pickle import loads
from pathlib import Path
from os import getenv

from openai import OpenAI
from openai.types import Batch

batches: list[Batch] = loads(Path("batches.pkl").read_bytes())
o = OpenAI(api_key=getenv("OPENAI_KEY"))

for i in batches:
  batch = o.batches.retrieve(i.id)
  if Path(f"embedding_results/{batch.id}.jsonl").exists():
    print("skip")
    continue
  if batch.status == "completed":
    print(f"{batch.id} completed")
    output_id = batch.output_file_id
    input_name = o.files.retrieve(batch.input_file_id).filename
    if output_id is None:
      print(f"{batch.id} has no output file")
      break
    file = o.files.content(output_id).read()
    print(f"write to {input_name}")
    _ = Path(f"embedding_results/{input_name}").write_bytes(file)
  elif batch.status == "failed":
    print(f"{batch.id} failed")
    break

