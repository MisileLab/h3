from pathlib import Path
from json import dumps
from uuid import uuid4

from polars import read_avro, col
from tqdm import tqdm

from utils import Data, read_cached_avro

comments = read_avro("comments.avro")
df = read_cached_avro("comments.avro")

Path("batches").mkdir(exist_ok=True)
prompt = Path("prompt").read_text()
comments_iter = comments.iter_rows(named=True)
batches: list[str] = []

def generate_image_urls(data: list[dict[str, dict[str, str] | str]], urls: list[str]):
  for url in urls:
    if url:
      data.append({
        "type": "image_url",
        "image_url": {
          "url": url
        }
      })
  return data

for k, i in tqdm(enumerate(comments.iter_rows(named=True))):
  data = Data.model_validate(i)
  if parent_id := data.parent_id:
    parent = Data.model_validate(comments.filter(col('comment_id') == parent_id).to_dicts()[0])
    parent_image_url = parent.author_image_url
    parent_string = dumps(parent.model_dump(exclude={"comment_id", "parent_id", "author_image_url", "video_id"}), ensure_ascii=False)
  else:
    parent_image_url = ""
    parent_string = ""
  current_image_url = data.author_image_url
  current_string = dumps(data.model_dump(exclude={"comment_id", "parent_id", "author_image_url", "video_id"}), ensure_ascii=False)
  batches.append(dumps(
    {
      "custom_id": data.comment_id,
      "method": "POST", 
      "url": "/v1/chat/completions",
      "body": {
        "model": "gpt-4.1-nano",
        "messages": [
          {"role": "system", "content": prompt},
          {"role": "user", "content": generate_image_urls([{
            "type": "text",
            "text": f"""
              first profile image is the current comment, second (if exist) is the parent comment.
              current comment: {current_string}
              parent comment: {parent_string}
              """
            }], [parent_image_url, current_image_url])
          }
        ]}
      }, ensure_ascii=False
    )
  )
  # if k != 0 and k % 9 == 0:
  _ = Path(f"batches/{uuid4()}.jsonl").write_text("\n".join(batches))
  batches = []

_ = Path(f"batches/{uuid4()}.json").write_text("\n".join(batches))

