from os import getenv

from openai import OpenAI

o = OpenAI(api_key=getenv("OPENAI_KEY"))

for i in o.files.list():
  print(i)
  _ = o.files.delete(i.id)

