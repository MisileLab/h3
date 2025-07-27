from os import getenv
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from inquirer import Editor, prompt # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]

_ = load_dotenv()

main_provider = OpenRouterProvider(
  api_key=getenv("OPENROUTER_API_KEY", "")
)

translate_provider = OpenAIProvider(
  api_key=getenv("OPENAI_API_KEY"),
  base_url=getenv("OPENAI_BASE_URL")
)

main_model = FallbackModel(
  OpenAIModel(
    model_name="moonshotai/kimi-k2:free",
    provider=main_provider
  ),
  OpenAIModel(
    model_name="moonshotai/kimi-k2",
    provider=main_provider
  )
)

translate_agent = Agent(
  model=OpenAIModel(
    model_name="Seed-X-PPO-7B",
    provider=translate_provider
  )
)

apply_character_agent = Agent(
  model=main_model,
  instructions=Path("./apply_character_prompt").read_text()
)

note_agent = Agent(
  model=main_model,
  instructions=Path("./note_prompt").read_text()
)

def verify[T](value: T | None) -> T:
  if value is None:
    raise ValueError("Value is None")
  return value

while True:
  original: str = verify(prompt(Editor("original")))["original"] # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
  note = note_agent.run_sync(f"<note>\n{Path('./note.txt').read_text()}\n</note>\n<story>\n{original}\n</story>").output
  _ = Path("./note.txt").write_text(note)
  for i in original.split("\n"): # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    print(i) # pyright: ignore[reportUnknownArgumentType]
    translated = translate_agent.run_sync(f"Translate the following Chinese sentence into Korean:\n{i}<ko>").output
    print(translated)
    _ = Path("./translated.txt").write_text(translated)
    applied = apply_character_agent.run_sync(f"<note>\n{Path('./note.txt').read_text()}\n</note>\n<translated>{translated}\n</translated><original>{i}\n</original>").output
    print(applied)
    _ = Path("./applied.txt").write_text(applied)
