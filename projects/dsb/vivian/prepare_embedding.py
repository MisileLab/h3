from pydantic import BaseModel
from tqdm import tqdm
from polars import read_avro, col, DataFrame, concat

from utils import ProcessedData

processed = read_avro("processed.avro")
df = DataFrame()

class EmbeddingData(BaseModel):
  parent_comment_author: str
  parent_comment_content: str
  comment_author: str
  comment_content: str
  is_bot_comment: bool

def append(df: DataFrame, data: EmbeddingData) -> DataFrame:
  return concat([df, DataFrame(data.model_dump())], how="vertical", rechunk=True)

for i in tqdm(processed.to_dicts()):
  data = ProcessedData.model_validate(i)
  parent_comment = processed.filter(col("comment_id") == data.parent_id).to_dicts()
  if len(parent_comment) == 0:
    parent_comment_author = ""
    parent_comment_content = ""
  else:
    parent_comment = ProcessedData.model_validate(parent_comment[0])
    parent_comment_author = parent_comment.author_name
    parent_comment_content = parent_comment.content
  df = append(df, EmbeddingData(
    parent_comment_author=parent_comment_author,
    parent_comment_content=parent_comment_content,
    comment_author=data.author_name,
    comment_content=data.content,
    is_bot_comment=data.is_bot_comment
  ))

df.write_avro("embedding_data.avro")

