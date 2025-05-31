from os import getenv
from pathlib import Path
from pickle import loads

import polars as pl
from dotenv import load_dotenv
from logfire import configure, instrument_openai
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

df = pl.read_csv('hf://datasets/basicv8vc/SimpleQA/simple_qa_test_set.csv')

_ = load_dotenv()

prompt = Path("./compare_prompt.txt").read_text()

# ========== Setup ==========
model = OpenAIModel(
  'gpt-4.1-nano',
  provider=OpenAIProvider(
    api_key=getenv('OPENAI_KEY')
  )
)

class TestResult(BaseModel):
  correct: bool | None

agent = Agent(
  model,
  model_settings=OpenAIModelSettings(temperature=0.0),
  instructions=prompt,
  output_type=TestResult
)

_ = configure(token=getenv('LOGFIRE_KEY'))
_ = instrument_openai()

Path("./cache").mkdir(exist_ok=True)

history = []

class Metadata(BaseModel):
  topic: str
  answer_type: str
  urls: list[str]

class Data(BaseModel):
  metadata: Metadata
  problem: str
  answer: str

def percentage(part: int, whole: int):
  return 100 * float(part)/float(whole)

df_test: list[str] = loads(Path("./test_result.pkl").read_bytes()) # pyright: ignore[reportAny]
answers: list[bool | None] = loads(Path("./test_answers.pkl").read_bytes()) # pyright: ignore[reportAny]

for generated, answer in zip(df_test, df.iter_rows(named=True)):
  data = Data.model_validate(answer)
  result = agent.run_sync(f"""
    original: {data.answer}
    generated: {generated}
  """).output
  _ = answers.append(result.correct)

print("correct:", percentage(answers.count(True), len(answers)))
print("incorrect:", percentage(answers.count(False), len(answers)))
print("unknown:", percentage(answers.count(None), len(answers)))

