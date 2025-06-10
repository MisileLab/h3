from pathlib import Path
from os import getenv
from time import sleep
from json import loads

from openai import OpenAI
from polars import DataFrame, read_avro, concat, col
from tqdm import tqdm

from utils import Data, ProcessedData

def append(df: DataFrame, data: ProcessedData) -> DataFrame:
  return concat([df, DataFrame(data.model_dump())], how="vertical", rechunk=True)

o = OpenAI(api_key=getenv('OPENAI_KEY'))
file_ids: list[str] = []
df = DataFrame()
comments = read_avro("comments.avro")

for i in tqdm(Path("./batches").glob("*.jsonl")):
  file_ids.append(
    o.files.create(
      file=i.open("rb"),
      purpose="batch"
    ).id
  )

batches: list[str] = []

for i in tqdm(file_ids):
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
      output = loads(o.files.content(output_file_id).text) # pyright: ignore[reportAny]
      comment_id: str = output["custom_id"] # pyright: ignore[reportAny]
      response: str = output['response']['body']['choices'][0]['message']['content'].strip() # pyright: ignore[reportAny]
      print(response)
      if response in ["A", "B"]:
        is_bot = response == "A"
        data = Data.model_validate(
          df.filter(col("comment_id") == comment_id).to_dicts()[0]
        )
        df = append(df, ProcessedData(
          is_bot_comment=is_bot, **data
        ))
      _ = o.files.delete(output_file_id)
      _ = o.files.delete(batch.input_file_id)
  sleep(60)

df.write_avro("processed.avro")

