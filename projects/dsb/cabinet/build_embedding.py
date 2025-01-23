from json import dumps
from pathlib import Path
from pickle import loads
from dataclasses import dataclass
from re import sub, compile

from twscrape import Tweet # pyright: ignore[reportMissingTypeStubs]
from tiktoken import get_encoding

enc = get_encoding("o200k_base") # gpt4o - https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
# test
assert enc.decode(enc.encode("hello world")) == "hello world"
url_filter = compile(r"(https?:\/\/)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)")

@dataclass
class Data:
  t: list[Tweet]
  suicidal: bool

data: dict[int, Data] = {}

for i in Path("results").glob("*.pkl"):
  data[int(i.name.strip(".pkl"))] = Data(loads(i.read_bytes()), True) # pyright: ignore[reportAny]

for i in Path("results_normal").glob("*.pkl"):
  data[int(i.name.strip(".pkl"))] = Data(loads(i.read_bytes()), False) # pyright: ignore[reportAny]

Path("embeddings").mkdir(exist_ok=True)

# exclude url, retweet, mention for now
for k, v in data.items():
  temp_data: list[Tweet] = []
  nxt_skip = False
  for i in v.t:
    if nxt_skip:
      nxt_skip = False
      continue
    if i.retweetedTweet is not None:
      print("retweeted")
      nxt_skip = True
      continue
    for mention in i.mentionedUsers:
      print(f"delete {mention.username}")
      i.rawContent = i.rawContent.replace(f"@{mention.username}", "").replace(f"@{mention.displayname}", "")
    i.rawContent = sub(url_filter, "", i.rawContent)
    temp_data.append(i)
      
for k, v in data.items():
  limit = 30000
  n = 0
  file_count = 0
  while n < len(v.t)-1:
    file_count += 1
    count = 0
    f = open(Path("embeddings", f"{'suicidal' if v.suicidal else 'normal'}{k}_{file_count}.jsonl"), "w")
    while n < len(v.t)-1:
      i = v.t[n]
      count += len(enc.encode(i.rawContent))
      if count > limit:
        print(f"split {file_count}")
        break
      _ = f.write(dumps({
        "custom_id": f"request-{n}",
        "method": "POST",
        "url": "/v1/embeddings",
        "body": {
          "model": "text-embedding-3-large",
          "input": i.rawContent
        }
      }, ensure_ascii=False) + "\n")
      n += 1

