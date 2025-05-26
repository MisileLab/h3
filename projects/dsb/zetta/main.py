from os import getenv
from pathlib import Path
from pickle import dumps, loads

import polars as pl
from dotenv import load_dotenv
from httpx import get
from logfire import configure, instrument_openai
from openai import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

df = pl.read_csv('hf://datasets/basicv8vc/SimpleQA/simple_qa_test_set.csv')

_ = load_dotenv()

prompt = """
Answer based on urls of pages (not other urls).
"""

# ========== Setup ==========
model = OpenAIModel(
  'gpt-4.1-nano',
  provider=OpenAIProvider(
    api_key=getenv('OPENAI_KEY')
  )
)

agent = Agent(model, model_settings=OpenAIModelSettings(temperature=0.0))

_ = configure(token=getenv('LOGFIRE_KEY'))
_ = instrument_openai()

@agent.system_prompt
def system_prompt():
  return prompt

history = []

@agent.tool # pyright: ignore[reportArgumentType]
async def get_page(ctx: RunContext[str], url: str) -> str:
  print(url)
  if url not in ctx.deps.split('\n'):
    return "This url doesn't allowed."
  return get(f"https://r.jina.ai/{url}", headers={
    "Authorization": f"Bearer {getenv('JINA_API_KEY')}",
    "X-Engine": "Browser",
    "Accept": "text/event-stream"
  }, timeout=20.0).raise_for_status().text

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
  df_test.append(agent.run_sync(f"""
  topic: {data.metadata.topic}
  answer_type: {data.metadata.answer_type}
  urls: {data.metadata.urls}
  question: {data.problem}
  """, message_history=[], deps='\n'.join(data.metadata.urls)).output) # pyright: ignore[reportArgumentType]
  _ = Path("test_result.pkl").write_bytes(dumps(df_test))

