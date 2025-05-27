import asyncio
from os import getenv
from pathlib import Path
from pickle import dumps, loads

import gradio as gr
from logfire import configure, instrument_openai
from puremagic import from_stream  # pyright: ignore[reportUnknownVariableType]
from pydantic_ai import Agent, BinaryContent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from httpx import HTTPError, post

from prompts import prompts
from tools.feedback import (
  tools as feedback_tools,  # pyright: ignore[reportUnknownVariableType]
)
from tools.memory import (
  tools as memory_tools,  # pyright: ignore[reportUnknownVariableType]
)

# ========== Setup ==========
model = OpenAIModel(
  'google/gemini-2.5-flash-preview',
  provider=OpenAIProvider(
    base_url='https://openrouter.ai/api/v1',
    api_key=getenv('OPENROUTER_KEY')
  )
)

web_model = OpenAIModel(
  'gpt-4.1-nano',
  provider=OpenAIProvider(api_key=getenv("OPENAI_KEY"))
)

agent = Agent(
  model,
  tools=[
    duckduckgo_search_tool(),
    *memory_tools,
    *feedback_tools
  ],  # pyright: ignore[reportUnknownArgumentType]
  mcp_servers=[
    MCPServerStdio(
      "pnpx",
      args=["mcp-neovim-server"],
      env={
        "ALLOW_SHELL_COMMANDS": "True",
        "NVIM_SOCKET_PATH": "/tmp/nvim"
      }
    )
  ]
)

web_agent = Agent(
  web_model,
  instructions=prompts["web"]
)

_ = configure(token=getenv('LOGFIRE_KEY'))
_ = instrument_openai()

@agent.system_prompt
def system_prompt():
  return prompts["main"]

@agent.tool_plain
async def get_page(query: str, url: str):
  """
  get page of url and get answer using LLM

  Args:
    query: query string that will be send to LLM
    url: url of page
  """
  resp = post("https://r.jina.ai", headers={
    "Authorization": f"Bearer {getenv('JINA_API_KEY')}",
    "X-Engine": "Browser",
    "Accept": "text/event-stream"
  }, data={
    "url": url
  }, timeout=None)
  try:
    resp = resp.raise_for_status()
  except HTTPError:
    return f"""
    status_code: {resp.status_code}
    text: {resp.text}
    """
  history = (await web_agent.run(resp.text, message_history=[])).all_messages()
  return await web_agent.run(query, message_history=history)

if Path("data.pkl").exists():
  initial = loads(Path("data.pkl").read_bytes()) # pyright: ignore[reportAny]
else:
  initial: tuple[list[ModelMessage], list[tuple[str, str]]] = ([], [])

history = initial[0]
chat_state = initial[1]

# ========== Async Chat Function ==========
async def respond(message: str, files: list[bytes]):
  bin_files: list[BinaryContent] = []
  if files:
    for f in files:
      media_type = from_stream(f, mime=True)
      bin_files.append(BinaryContent(data=f, media_type=media_type))

  async with agent.run_mcp_servers():
    response = await agent.run([message, *bin_files], message_history=history)
    history.clear()
    history.extend(response.all_messages())
    chat_state.append((message, response.output))
    return chat_state

# ========== Gradio Interface ==========
with gr.Blocks() as demo:
  chatbot = gr.Chatbot()
  msg = gr.Textbox(label="Message", placeholder="Type your message and press Enter")
  file_upload = gr.File(label="Upload files", file_count="multiple", file_types=["file"])
  
  def save():
    _ = Path("data.pkl").write_bytes(dumps([history, chat_state])) 

  with gr.Row():
    send_btn = gr.Button("Send")
    undo_btn = gr.Button("↩️ Undo")
    reset_btn = gr.Button("🔄 Reset")

  # Send button logic
  def sync_respond(message: str, files: list[bytes]):
    return asyncio.run(respond(message, files))

  _ = send_btn.click(
    fn=sync_respond,
    inputs=[msg, file_upload],
    outputs=[chatbot]
  ).then(
    fn=lambda: "",
    inputs=None,
    outputs=msg
  ).then(
    fn=save,
    inputs=[],
    outputs=None
  )

  # Undo button logic
  def undo():
    if len(chat_state) >= 1:
      _ = history.pop()
      _ = history.pop()
      _ = chat_state.pop()
    return chat_state

  _ = undo_btn.click(
    fn=undo,
    inputs=[],
    outputs=[chatbot]
  ).then(
    fn=save,
    inputs=[],
    outputs=None
  )

  # Reset button logic
  def reset_all() -> tuple[list[str], None]:
    history.clear()
    chat_state.clear()
    return [], None

  _ = reset_btn.click(
    fn=reset_all,
    inputs=[],
    outputs=[chatbot, file_upload]
  ).then(
    fn=save,
    inputs=[],
    outputs=None
  )

  _ = demo.load(fn=lambda: chat_state, outputs=[chatbot])

if __name__ == "__main__":
  _ = demo.launch()

