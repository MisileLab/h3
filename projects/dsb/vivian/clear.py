from os import getenv

from openai import OpenAI
from tqdm import tqdm

o = OpenAI(api_key=getenv("OPENAI_KEY"))

for i in tqdm(list(o.files.list())):
  _ = o.files.delete(i.id)

