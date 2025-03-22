from json import dumps
from lib import Data

from polars import read_avro

a = read_avro("data.avro")
prompt = ""

with open("data.jsonl", "w") as f:
  for i in a.to_dicts():
    val = Data.model_validate(i)
    messages = [{"role": "system", "content": prompt}]
    for conv, analysis in zip(val.conversations, val.analysis):
      messages.append({"role": "user", "content": conv})
      messages.append({
        "role": "assistant",
        "content": f"# Analysis\n{analysis}\n# Conclusion\nsuicidal: {val.suicidal}"
      })
    _ = f.write(dumps(messages) + "\n")

