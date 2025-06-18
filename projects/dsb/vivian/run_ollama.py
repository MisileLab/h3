from pathlib import Path
from json import dumps

from pydantic_ai import Agent
from polars import read_avro, col, DataFrame, concat
from tqdm import tqdm

from utils import Data, read_cached_avro, ProcessedData

comments = read_avro("comments.avro")
df = read_cached_avro("processed.avro")

prompt = Path("prompt").read_text()
comments_iter = comments.iter_rows(named=True)

# def generate_image_urls(data: list[dict[str, dict[str, str] | str]], urls: list[str]):
#   for url in urls:
#     if url:
#       data.append({
#         "type": "image_url",
#         "image_url": {
#           "url": url
#         }
#       })
#   return data

def append(df: DataFrame, data: ProcessedData) -> DataFrame:
  return concat([df, DataFrame(data.model_dump())], how="vertical", rechunk=True)

agent = Agent(
  'ollama:phi4-reasoning:plus',
  system_prompt=prompt
)

for k, i in tqdm(enumerate(comments.iter_rows(named=True))):
  data = Data.model_validate(i)
  if parent_id := data.parent_id:
    parent = Data.model_validate(comments.filter(col('comment_id') == parent_id).to_dicts()[0])
    #parent_image_url = parent.author_image_url
    parent_string = dumps(parent.model_dump(exclude={"comment_id", "parent_id", "author_image_url", "video_id"}), ensure_ascii=False)
  else:
    #parent_image_url = ""
    parent_string = ""
  #current_image_url = data.author_image_url
  current_string = dumps(data.model_dump(exclude={"comment_id", "parent_id", "author_image_url", "video_id"}), ensure_ascii=False)
  response = agent.run_sync("""first profile image is the current comment, second (if exist) is the parent comment.
  current comment: {current_string}
  parent comment: {parent_string}""").output
  if response in ["A", "B"]:
    is_bot = response == "A"
    df = append(df, ProcessedData(
      is_bot_comment=is_bot, **data.model_dump() # pyright: ignore[reportAny]
    ))
    df.write_avro("processed.avro")

