from os import getenv
from pathlib import Path
from time import sleep

import polars as pl
from blake3 import blake3
from dotenv import load_dotenv
from httpx import AsyncClient
from logfire import configure, instrument_openai
from pydantic import BaseModel
from pydantic_ai import Agent, ModelHTTPError, RunContext
from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider

df = pl.read_csv('hf://datasets/basicv8vc/SimpleQA/simple_qa_test_set.csv')

_ = load_dotenv()

prompt = Path("./prompt.txt").read_text()
summarize_prompt = Path("./summarize_prompt.txt").read_text()

provider = OpenRouterProvider(api_key=getenv('OPENROUTER_KEY', ''))
setting = OpenAIModelSettings(temperature=0.0)

model = OpenAIModel(
  model_name=getenv('MODEL_NAME_PAID', ''),
  provider=provider
)

# ========== Setup ==========
agent = Agent(
  model,
  model_settings=setting,
  instructions=prompt,
  deps_type=str
)

summarize_agent = Agent(
  model,
  model_settings=setting,
  instructions=summarize_prompt
)

_ = configure(token=getenv('LOGFIRE_KEY'))
_ = instrument_openai()

Path("./cache/jinaai").mkdir(exist_ok=True, parents=True)
Path("./cache/summarized").mkdir(exist_ok=True, parents=True)

history = []

async def get_jina_page(url: str) -> str:
  cache = Path(f"./cache/jinaai/{blake3(url.encode()).hexdigest()}")
  if cache.exists():
    return cache.read_text()
  resp = ""
  async with AsyncClient(timeout=None) as client:
    async with client.stream("POST", "https://r.jina.ai", headers={
      "Authorization": f"Bearer {getenv('JINA_API_KEY')}",
      "X-Engine": "Browser",
      "Accept": "text/event-stream",
      # "X-Respond-With": "readerlm-v2"
      }, data={
        "url": url
      }, timeout=None) as response:
        async for line in response.aiter_lines():
          if line.startswith("data:"):
            data_line = line.removeprefix("data:").strip()
            resp = data_line 
  _ = cache.write_text(resp)
  return resp

# Let's try without summarizing first, since we fixed jina.ai
@agent.tool
async def get_page(ctx: RunContext[str], url: str) -> str:
  print(url)
  if url not in ctx.deps.split('\n'):
    return "This url doesn't allowed."
  cache = Path(f"./cache/summarized/{blake3(url.encode()).hexdigest()}")
  if cache.exists():
    return cache.read_text()
  resp = await get_jina_page(url)
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
  _ = cache.write_text(resp)
  return resp

@agent.tool
async def get_full_page(ctx: RunContext[str], url: str) -> str:
  print(url)
  if url not in ctx.deps.split('\n'):
    return "This url doesn't allowed."
  return await get_jina_page(url)

class Metadata(BaseModel):
  topic: str
  answer_type: str
  urls: list[str]

class Data(BaseModel):
  metadata: Metadata
  problem: str
  answer: str

df_test = pl.DataFrame() if not Path("zetta.avro").exists() else pl.read_avro("zetta.avro")

def append(df: pl.DataFrame, question: str, answer: str, generated: str) -> pl.DataFrame:
  return pl.concat([df, pl.DataFrame({
    "question": question,
    "answer": answer,
    "generated": generated
  })], rechunk=True)

def evaluate(df: pl.DataFrame, data: Data) -> pl.DataFrame:
  return append(df, data.problem, data.answer, agent.run_sync(f"""
    topic: {data.metadata.topic}
    answer_type: {data.metadata.answer_type}
    urls: {data.metadata.urls}
    question: {data.problem}
  """, message_history=[], deps='\n'.join(data.metadata.urls)).output)

for i in df.iter_rows(named=True):
  i["metadata"] = eval(i["metadata"]) # pyright: ignore[reportAny]
  data = Data.model_validate(i)
  if (
    len(df_test) > 0 and
    df_test.filter(pl.col("question") == data.problem).shape[0] > 0 # pyright: ignore[reportUnknownMemberType]
  ):
    continue
  try:
    df_test = evaluate(df_test, data)
  except ModelHTTPError as e:
    if e.status_code == 429:
      print("Rate limit exceeded, waiting for 60 seconds...")
      print(e.body)
      sleep(60)
      df_test = evaluate(df_test, data)
    else:
      raise e
  df_test.write_avro("zetta.avro")

