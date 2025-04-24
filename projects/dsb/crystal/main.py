from asyncio import run
from typing import override
from pathlib import Path

from httpx import get
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.messages import ModelMessage
from rich.console import Console, ConsoleOptions, RenderResult
from rich.markdown import CodeBlock, Markdown
from rich.syntax import Syntax
from rich.text import Text

agent = Agent(
  'google-gla:gemini-2.5-pro-exp-03-25',
  system_prompt = (Path("./prompt").read_text()),
  tools = [
    duckduckgo_search_tool()
  ]
)

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
    inp = input("user: ")
    response = await agent.run(inp, message_history=history)
    history = response.all_messages()
    console.print(Markdown(response.output))
    console.print(response.usage())
# TODO: https://www.tomsguide.com/ai/these-5-ai-prompts-work-like-magic-no-matter-which-chatbot-you-use

if __name__ == "__main__":
  run(main())

