from asyncio import run
from json import loads
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from tools.data import functions as data_functions, get_current_status

class Config(BaseModel):
  openai_url: str
  openai_key: str
  casual_model: str
  high_reasoning_model: str

config = Config.model_validate(loads(Path("./config.json").read_text()))

model = OpenAIModel(
  provider=OpenAIProvider(
    base_url=config.openai_url,
    api_key=config.openai_key
  ),
  model_name=config.casual_model
)

agent = Agent(
  model=model,
  tools=[
    *data_functions,
  ],
  instructions=Path("./prompts/scalar").read_text()
)

async def main():
  console = Console()
  while True:
    async with agent.run_stream(f"Crystal: {input("Crystal: ").strip()}" + f"\n{get_current_status()}") as result:
      with Live('', console=console, vertical_overflow="visible") as live:
        async for message in result.stream_text():
          live.update(Markdown(message))
        console.log(result.usage())

if __name__ == "__main__":
  run(main())

