from asyncio import run
from typing import override
from pathlib import Path
from os import getenv
from puremagic import from_file # pyright: ignore[reportUnknownVariableType]

from httpx import get
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.messages import ModelMessage
from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import CodeBlock, Markdown
from rich.syntax import Syntax
from rich.text import Text
from inquirer import prompt, Editor # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from logfire import configure, instrument_openai

from tools.memory import tools as memory_tools # pyright: ignore[reportUnknownVariableType]

model = OpenAIModel(
  'google/gemini-2.5-flash-preview',
  provider=OpenAIProvider(
    base_url='https://openrouter.ai/api/v1',
    api_key=getenv('OPENROUTER_KEY')
  )
)

agent = Agent(
  model,
  system_prompt = (Path("./prompt").read_text().replace('<Username />', getenv("USER_ID", "misile"))),
  tools = [
    duckduckgo_search_tool(),
    *memory_tools
  ] # pyright: ignore[reportUnknownArgumentType]
)

_ = configure(token=getenv('LOGFIRE_KEY'))
_ = instrument_openai()

console = Console()

@agent.tool_plain
async def request(url: str) -> str:
  """
  request to url and get text
  """
  console.print(f'request {url}', style='blue')
  page = get(url)
  console.print(page.status_code, style='magenta')
  return page.text if not page.is_error else f"status: {page.status_code}, text: {page.text}"

def prettier_code_blocks():
  """Make rich code blocks prettier and easier to copy.

  From https://github.com/samuelcolvin/aicli/blob/v0.8.0/samuelcolvin_aicli.py#L22
  """

  class SimpleCodeBlock(CodeBlock):
    @override
    def __rich_console__(
      self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
      code = str(self.text).rstrip()
      yield Text(self.lexer_name, style='dim')
      yield Syntax(
        code,
        self.lexer_name,
        theme=self.theme,
        background_color='default',
        word_wrap=True,
      )
      yield Text(f'/{self.lexer_name}', style='dim')

  Markdown.elements['fence'] = SimpleCodeBlock

async def main():
  prettier_code_blocks()
  history: list[ModelMessage] = []
  while True:
    _inp: dict[str, str] | None = prompt( # pyright: ignore[reportUnknownVariableType]
      [
        Editor("input", message="user"),
        Editor("files", message="files (seperated by newline)")
      ]
    )
    if _inp is None:
      exit()
    inp = _inp["input"]
    files: list[BinaryContent] = []
    if not _inp["files"] == "":
      for i in _inp["files"].split("\n"):
        p = Path(i)
        _ = files.append(BinaryContent(data=p.read_bytes(), media_type=from_file(p, True)))
    response = await agent.run([
      inp,
      *files
    ], message_history=history)
    history = response.all_messages()
    console.print(Markdown(response.output))
    console.print(response.usage())
# TODO: https://www.tomsguide.com/ai/these-5-ai-prompts-work-like-magic-no-matter-which-chatbot-you-use

if __name__ == "__main__":
  run(main())

