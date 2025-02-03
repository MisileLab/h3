from json import dumps
from pathlib import Path

from tiktoken import get_encoding

from lib import read_pickle, Data

enc = get_encoding("o200k_base") # gpt4o - https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
# test
assert enc.decode(enc.encode("hello world")) == "hello world"

data = read_pickle("data.pkl")

Path("embeddings").mkdir(exist_ok=True)

for _i in data.to_dict('records'): # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
  i = Data.model_validate(_i)
  limit = 30000
  n = 0
  file_count = 0
  while n < len(i.data)-1:
    file_count += 1
    count = 0
    with open(Path("embeddings", f"{'suicidal' if i.suicidal else 'normal'}{i.uid}_{file_count}.jsonl"), "w") as f:
      while n < len(i.data)-1:
        d = i.data[n]
        count += len(enc.encode(d))
        if count > limit:
          print(f"split {file_count}")
          break
        _ = f.write(dumps({
          "custom_id": f"request-{n}",
          "method": "POST",
          "url": "/v1/embeddings",
          "body": {
            "model": "text-embedding-3-large",
            "input": d
          }
        }, ensure_ascii=False) + "\n")
        n += 1

