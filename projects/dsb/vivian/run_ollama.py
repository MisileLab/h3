from pathlib import Path
from json import dumps
from types import CoroutineType
import asyncio
from itertools import islice

from pydantic_ai import Agent
from polars import read_avro, col, DataFrame, concat
from tqdm import tqdm
from torch import device as Device
from torch.cuda import get_device_properties, is_available # pyright: ignore[reportUnknownVariableType]

from utils import Data, read_cached_avro, ProcessedData

comments = read_avro("comments.avro")
df = read_cached_avro("processed.avro")

prompt = Path("prompt").read_text()
comments_iter = comments.iter_rows(named=True)

def append(df: DataFrame, data: ProcessedData) -> DataFrame:
  return concat([df, DataFrame(data.model_dump())], how="vertical", rechunk=True)

def get_batch_size() -> int:
  if not is_available():
    return 2

  device = Device('cuda:0')
  total_memory: float = get_device_properties(device).total_memory / (1024**3) # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
  return int(total_memory // 11) # pyright: ignore[reportUnknownArgumentType]

async def process_comment(agent: Agent, data: Data, comments_df: DataFrame) -> ProcessedData | None:
  if parent_id := data.parent_id:
    parent = Data.model_validate(comments_df.filter(col('comment_id') == parent_id).to_dicts()[0])
    parent_string = dumps(parent.model_dump(exclude={"comment_id", "parent_id", "author_image_url", "video_id"}), ensure_ascii=False)
  else:
    parent_string = "" 
  current_string = dumps(data.model_dump(exclude={"comment_id", "parent_id", "author_image_url", "video_id"}), ensure_ascii=False)
  response = await agent.run(f"""first profile image is the current comment, second (if exist) is the parent comment.
  current comment: {current_string}
  parent comment: {parent_string}""")

  if response.output in ["A", "B"]:
    return ProcessedData(
      is_bot_comment=response.output == "A",
      **data.model_dump() # pyright: ignore[reportAny]
    )
  return None

async def process_batch(batch: list[dict[str, str]], agent: Agent, comments_df: DataFrame) -> list[ProcessedData]:
  tasks: list[CoroutineType[None, None, ProcessedData | None]] = []
  for item in batch:
    data = Data.model_validate(item)
    tasks.append(process_comment(agent, data, comments_df))

  results = await asyncio.gather(*tasks)
  return [r for r in results if r is not None]

async def main():
  agent = Agent(
    'ollama:phi4-reasoning:plus',
    system_prompt=prompt
  )

  batch_size = get_batch_size()
  df = read_cached_avro("processed.avro")

  with tqdm(total=len(comments)) as pbar:
    while True:
      batch = list(islice(comments_iter, batch_size))
      if not batch:
        break

      processed_batch = await process_batch(batch, agent, comments)
      for processed in processed_batch:
        df = append(df, processed)

      df.write_avro("processed.avro")
      _ = pbar.update(len(batch))

if __name__ == "__main__":
  asyncio.run(main())
