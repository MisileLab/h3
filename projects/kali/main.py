from os import getenv
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
import logfire

_ = load_dotenv()
_ = logfire.configure(token=getenv('LOGFIRE_KEY', ''))
_ = logfire.instrument_pydantic_ai()

main_provider = OpenRouterProvider(
  api_key=getenv("OPENROUTER_API_KEY", "")
)

# translate_provider = OpenAIProvider(
#   api_key=getenv("OPENAI_API_KEY"),
#   base_url=getenv("OPENAI_BASE_URL")
# )

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

# translate_agent = Agent(
#   model=OpenAIModel(
#     model_name="Seed-X-PPO-7B",
#     provider=translate_provider
#   )
# )

translate_agent = Agent(
  model=main_model,
  instructions=Path("./translate_prompt").read_text()
)

apply_character_agent = Agent(
  model=main_model,
  instructions=Path("./apply_character_prompt").read_text()
)

note_agent = Agent(
  model=main_model,
  instructions=Path("./note_prompt").read_text()
)

while True:
  prev_original: str = Path("./original_prev.txt").read_text()
  prev_translated: str = Path("./translated_prev.txt").read_text()
  original: str = Path("./original.txt").read_text()
  note = note_agent.run_sync(f"""
    <note>
      {Path('./note.txt').read_text()}
    </note>
    <story>
      {original}
    </story>
    <prev>
      {prev_original}
    </prev>
    <translated>
      {prev_translated}
    </translated>""").output
  _ = Path("./note.txt").write_text(data=note)
  for i in original.split("\n"):
    print(i)
    translated = translate_agent.run_sync(f"Translate the following English sentence into Korean:\n{i}<ko>").output
    print(translated)
    _ = Path("./translated.txt").write_text(translated)
    applied = apply_character_agent.run_sync(f"<note>\n{Path('./note.txt').read_text()}\n</note>\n<translated>{translated}\n</translated><original>{i}\n</original>").output
    print(applied)
    _ = Path("./applied.txt").write_text(applied)
