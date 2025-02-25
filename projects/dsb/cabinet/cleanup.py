from pickle import loads
from pathlib import Path
from os import getenv

from openai import OpenAI
from openai.types import Batch

batches: list[Batch] = loads(Path("batches.pkl").read_bytes())
o = OpenAI(api_key=getenv("OPENAI_KEY"))

for i in batches:
  batch = o.batches.retrieve(i.id)
  if batch.status == "completed":
    print(f"{batch.id} completed")
    input_id = batch.input_file_id
    output_id = batch.output_file_id
    input_name = o.files.retrieve(input_id).filename
    if Path(f"embedding_results/{input_name}").is_file():
      _ = o.files.delete(input_id)
      if output_id is not None:
        _ = o.files.delete(output_id)

