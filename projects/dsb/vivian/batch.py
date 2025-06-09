from pathlib import Path
from os import getenv
from time import sleep

from openai import OpenAI
from polars import DataFrame, read_avro, concat

from utils import ProcessedData

def append(df: DataFrame, data: ProcessedData) -> DataFrame:
  return concat([df, DataFrame(data.model_dump())], how="vertical", rechunk=True)

o = OpenAI(api_key=getenv('OPENAI_KEY'))
file_ids: list[str] = []
df = DataFrame()
comments = read_avro("comments.avro")

for i in Path("./batches").glob("*.jsonl"):
  file_ids.append(
    o.files.create(
      file=i.open("rb"),
      purpose="batch"
    ).id
  )

batches: list[str] = []

for i in file_ids:
  batches.append(o.batches.create(
    completion_window='24h',
    endpoint='/v1/chat/completions',
    input_file_id=i
  ).id)

while batches:
  will_delete = []
  for i in batches:
    batch = o.batches.retrieve(i)
    print(batch.status)
    output_file_id = batch.output_file_id
    if batch.status == "completed" and output_file_id is not None:
      output = o.files.content(output_file_id).text
      # TODO: split by '\n' and add data to df
  sleep(60)

df.write_avro("processed.avro")

