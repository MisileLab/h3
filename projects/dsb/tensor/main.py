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
from pydantic_ai.messages import (
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    TextPartDelta,
    ToolCallPartDelta,
)

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
    tool_name = ""
    async with agent.iter(f"Crystal: {input('Crystal: ').strip()}" + f"\n{get_current_status()}") as result:
        with Live('', console=console, vertical_overflow="visible") as live:
            async for node in result:
                if Agent.is_user_prompt_node(node):
                    live.update(Markdown(f'User: {node.user_prompt}'))
                elif Agent.is_model_request_node(node):
                    async with node.stream(result.ctx) as stream:
                        final_result_found = False
                        async for event in stream:
                            if isinstance(event, FinalResultEvent):
                                final_result_found = True
                                break
                        if final_result_found:
                            async for output in stream.stream_text():
                                live.update(Markdown(f'Final Response: {output}'))
                elif Agent.is_call_tools_node(node):
                    async with node.stream(result.ctx) as stream:
                        async for event in stream:
                            if isinstance(event, FunctionToolCallEvent):
                                tool_name = event.part.tool_name
                                live.update(Markdown(f"Calling tool {tool_name} with args {event.part.args}"))
                            elif isinstance(event, FunctionToolResultEvent):
                                live.update(Markdown(f"Tool {tool_name} finished with output {event.result.content}"))

if __name__ == "__main__":
  run(main())
