from os import getenv
from pathlib import Path
from pickle import dumps, loads
from time import sleep

import polars as pl
from blake3 import blake3
from dotenv import load_dotenv
from httpx import post
from logfire import configure, instrument_openai
from pydantic import BaseModel
from pydantic_ai import Agent, ModelHTTPError, RunContext
from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

df = pl.read_csv('hf://datasets/basicv8vc/SimpleQA/simple_qa_test_set.csv')

_ = load_dotenv()

prompt = Path("./prompt.txt").read_text()
summarize_prompt = Path("./summarize_prompt.txt").read_text()

# ========== Setup ==========
model = OpenAIModel(
  'gpt-4.1-nano',
  provider=OpenAIProvider(
    api_key=getenv('OPENAI_KEY')
  )
)

summarize_model = OpenAIModel(
  'meta-llama/llama-4-scout',
  provider=OpenAIProvider(
    api_key=getenv('OPENROUTER_KEY'),
    base_url='https://openrouter.ai/api/v1'
  )
)

agent = Agent(
  model,
  model_settings=OpenAIModelSettings(temperature=0.0),
  instructions=prompt,
  deps_type=str
)

summarize_agent = Agent(
  summarize_model,
  model_settings=OpenAIModelSettings(temperature=0.0),
  instructions=summarize_prompt
)

_ = configure(token=getenv('LOGFIRE_KEY'))
_ = instrument_openai()

Path("./cache").mkdir(exist_ok=True)

history = []

@agent.tool
async def get_page(ctx: RunContext[str], url: str) -> str:
  print(url)
  if url not in ctx.deps.split('\n'):
    return "This url doesn't allowed."
  blake3_hash = blake3(url.encode()).hexdigest()
  if Path(f"./cache/{blake3_hash}").exists():
    return Path(f"./cache/{blake3_hash}").read_text()
  resp = post("https://r.jina.ai", headers={
    "Authorization": f"Bearer {getenv('JINA_API_KEY')}",
    "X-Engine": "Browser",
    "Accept": "text/event-stream"
  }, data={
    "url": url
  }, timeout=None).raise_for_status().text
  try:
    resp = (await summarize_agent.run(resp, message_history=[])).output
  except ModelHTTPError as e:
    if e.status_code == 429:
      print("Rate limit exceeded, waiting for 60 seconds...")
      print(e.body)
      sleep(60)
      resp = (await summarize_agent.run(resp, message_history=[])).output
    else:
      raise e
  _ = Path(f"./cache/{blake3_hash}").write_text(resp)
  return resp

class Metadata(BaseModel):
  topic: str
  answer_type: str
  urls: list[str]

class Data(BaseModel):
  metadata: Metadata
  problem: str
  answer: str

df_test: list[str] = loads(Path("./test_result.pkl").read_bytes()) if Path("test_result.pkl").exists() else []
j = 1

for i in df.iter_rows(named=True):
  if len(df_test) >= j:
    j += 1
    continue
  i["metadata"] = eval(i["metadata"]) # pyright: ignore[reportAny]
  data = Data.model_validate(i)
  try:
    df_test.append(agent.run_sync(f"""
    topic: {data.metadata.topic}
    answer_type: {data.metadata.answer_type}
    urls: {data.metadata.urls}
    question: {data.problem}
    """, message_history=[], deps='\n'.join(data.metadata.urls)).output)
  except ModelHTTPError as e:
    if e.status_code == 429:
      print("Rate limit exceeded, waiting for 60 seconds...")
      print(e.body)
      sleep(60)
      df_test.append(agent.run_sync(f"""
      topic: {data.metadata.topic}
      answer_type: {data.metadata.answer_type}
      urls: {data.metadata.urls}
      question: {data.problem}
      """, message_history=[], deps='\n'.join(data.metadata.urls)).output)
    else:
      raise e
  _ = Path("test_result.pkl").write_bytes(dumps(df_test))

