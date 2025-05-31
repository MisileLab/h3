from os import getenv
from pathlib import Path
from pickle import dumps

import polars as pl
from dotenv import load_dotenv
from logfire import configure, instrument_openai
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

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
  reasoning: str

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

class Data(BaseModel):
  question: str
  answer: str
  generated: str

def percentage(part: int, whole: int):
  return 100 * float(part)/float(whole)

df = pl.read_avro("./zetta.avro")
answers: list[bool | None] = []

for i in df.iter_rows(named=True):
  data = Data.model_validate(i)
  result = agent.run_sync(f"""
    question: {data.question}
    original: {data.answer}
    generated: {data.generated}
  """).output
  _ = answers.append(result.correct)

_ = Path("./test.pkl").write_bytes(dumps(answers))

print("correct:", percentage(answers.count(True), len(answers)))
print("incorrect:", percentage(answers.count(False), len(answers)))
print("unknown:", percentage(answers.count(None), len(answers)))

