from pathlib import Path

from polars import Boolean, String, read_avro, DataFrame
from pydantic import BaseModel


def read_cached_avro(file_path: str):
  return read_avro(file_path) if Path(file_path).exists() else DataFrame()

class Data(BaseModel):
  comment_id: str
  content: str
  author_name: str
  author_image_url: str
  video_id: str
  parent_id: str = ""

class ProcessedData(Data):
  is_bot_comment: bool

schema = {
  "comment_id": String,
  "content": String,
  "author_name": String,
  "author_image_url": String,
  "video_id": String,
  "parent_id": String
}

processed_schema = schema | {
  "is_bot_comment": Boolean
}

