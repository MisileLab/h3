from os import getenv

from openai import OpenAI
from tqdm import tqdm

o = OpenAI(api_key=getenv("OPENAI_KEY"))

for i in tqdm(o.files.list()):
  print(i)
  _ = o.files.delete(i.id)

