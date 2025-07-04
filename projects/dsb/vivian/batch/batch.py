from pathlib import Path
from os import getenv
from time import sleep
from json import loads

from openai import OpenAI
from polars import DataFrame, read_avro, concat, col
from tqdm import tqdm

from utils import Data, ProcessedData, read_cached_avro

def append(df: DataFrame, data: ProcessedData) -> DataFrame:
  return concat([df, DataFrame(data.model_dump())], how="vertical", rechunk=True)

o = OpenAI(api_key=getenv('OPENAI_KEY'))
comments = read_avro("comments.avro")
df = read_cached_avro("processed.avro")
t = list(Path("./batches").glob("*.jsonl"))
for _ in range(len(df.to_dicts())):
  t = t[1:]

for i in (progress_bar := tqdm(t)):
  file_id = o.files.create(
    file=i.open("rb"),
    purpose="batch"
  ).id
  batch_id = o.batches.create(
    completion_window='24h',
    endpoint='/v1/chat/completions',
    input_file_id=file_id
  ).id
  batch = o.batches.retrieve(batch_id)
  while batch.status != "completed":
    progress_bar.set_description_str(batch.status)
    sleep(1)
    batch = o.batches.retrieve(batch_id)
    if batch.status == "failed":
      errors = batch.errors
      if errors is not None:
        data = errors.data
        if batch.status == "failed" and data is not None and data[0].code == "token_limit_exceeded":
          progress_bar.set_description_str('ratelimit hitted')
          sleep(60 * 20)
          batch_id = o.batches.create(
            completion_window='24h',
            endpoint='/v1/chat/completions',
            input_file_id=file_id
          ).id
  output_file_id = batch.output_file_id
  if output_file_id is not None:
    outputs = [
      loads(i)
      for i in o.files.content(output_file_id).text.removesuffix('\n').split("\n")
    ]
    for output in outputs: # pyright: ignore[reportAny]
      comment_id: str = output["custom_id"] # pyright: ignore[reportAny]
      response: str = output['response']['body']['choices'][0]['message']['content'].strip() # pyright: ignore[reportAny]
      if response in ["A", "B"]:
        is_bot = response == "A"
        data = Data.model_validate(
          comments.filter(col("comment_id") == comment_id).to_dicts()[0]
        ).model_dump()
        df = append(df, ProcessedData(
          is_bot_comment=is_bot, **data # pyright: ignore[reportAny]
        ))
    _ = o.files.delete(output_file_id)
    _ = o.files.delete(batch.input_file_id)
    df.write_avro("processed.avro")
  else:
    print("something wrong")
    exit(1)

