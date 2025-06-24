#!pip install -U transformers sentence-transformers tqdm pydantic polars accelerate
#!apt install zstd
#!zstd -d --rm embedding_data.avro.zst

from pydantic import BaseModel
from tqdm import tqdm
from polars import read_avro, DataFrame, concat
from sentence_transformers import SentenceTransformer

processed = read_avro("embedding_data.avro")
df = DataFrame()

# class EmbeddingData(BaseModel):
#   parent_comment_author: str
#   parent_comment_content: str
#   comment_author: str
#   comment_content: str
#   is_bot_comment: bool

class Embedding(BaseModel):
  parent_comment_author: list[float]
  parent_comment_content: list[float]
  comment_author: list[float]
  comment_content: list[float]
  is_bot_comment: int

model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")

def append(df: DataFrame, data: Embedding) -> DataFrame:
  return concat([df, DataFrame(data.model_dump())], how="vertical", rechunk=True)

for i in tqdm(processed.to_dicts()):
  is_bot = 1 if i["is_bot_comment"] else 0
  del i["is_bot_comment"]
  encoded = model.encode([ # pyright: ignore[reportUnknownMemberType]
    i["parent_comment_author"],
    i["parent_comment_content"],
    i["comment_author"],
    i["comment_content"]
  ])
  df = append(df, Embedding(
    parent_comment_author=encoded[0], # pyright: ignore[reportAny]
    parent_comment_content=encoded[1], # pyright: ignore[reportAny]
    comment_author=encoded[2], # pyright: ignore[reportAny]
    comment_content=encoded[3], # pyright: ignore[reportAny]
    is_bot_comment=is_bot
  ))

df.write_avro("embedding.avro")

