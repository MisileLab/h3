from json import dumps
from os import getenv
from pathlib import Path
from typing import Callable

from prompts import prompts
from pydantic_ai import Agent

user_id = getenv("USER_ID", "misile")

agent = Agent(
  "openai:gpt-4.1-mini"
)

def get_current_prompt():
  return prompts["main"]

async def modify_prompt(
  actual_input: str,
  actual_output: str,
  desired_output: str
):
  """
  change prompt based on actual input, output, and desired output by LLM
  returns original prompt (not changed)
  """
  main_prompt = prompts["main"]
  user_input: dict[str, str] = {
    "actual_input": actual_input,
    "actual_output": actual_output,
    "desired_output": desired_output,
    "current_prompt": get_current_prompt()
  }
  prompts["main"] = (await agent.run(dumps(user_input))).output
  _ = Path("./prompts/main").write_text(prompts["main"])
  return main_prompt

tools: list[Callable] = [get_current_prompt, modify_prompt] # pyright: ignore[reportMissingTypeArgument, reportUnknownVariableType]

