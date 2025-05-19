import gradio as gr
import asyncio
import json
from os import getenv, path
from puremagic import from_stream  # pyright: ignore[reportUnknownVariableType]

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.messages import ModelMessage
from pydantic_ai.mcp import MCPServerStdio

from tools.memory import tools as memory_tools  # pyright: ignore[reportUnknownVariableType]
from tools.feedback import tools as feedback_tools  # pyright: ignore[reportUnknownVariableType]
from prompts import prompts

from logfire import configure, instrument_openai

# Path to store chat history
HISTORY_FILE = "chat_history.json"

# ========== Persistence Helpers ==========
def load_history_from_disk() -> list[tuple[str, str]]:
  if path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
      return json.load(f) # pyright: ignore[reportAny]
  return []

def save_history_to_disk(chat: list[tuple[str, str]]):
  with open(HISTORY_FILE, "w", encoding="utf-8") as f:
    json.dump(chat, f, ensure_ascii=False, indent=2)

# ========== Setup ==========
model = OpenAIModel(
  'google/gemini-2.5-flash-preview',
  provider=OpenAIProvider(
    base_url='https://openrouter.ai/api/v1',
    api_key=getenv('OPENROUTER_KEY')
  )
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

_ = configure(token=getenv('LOGFIRE_KEY'))
_ = instrument_openai()

@agent.system_prompt
def system_prompt():
  return prompts["main"]

# Load persisted chat state
initial_chat = load_history_from_disk()
history: list[ModelMessage] = []  # used for agent context

# ========== Async Chat Function ==========
async def respond(message: str, files: list[bytes], chat: list[tuple[str, str]]):
  # Append user message
  chat.append((message, None))
  save_history_to_disk(chat)

  bin_files: list[BinaryContent] = []
  if files:
    for f in files:
      media_type = from_stream(f, mime=True)
      bin_files.append(BinaryContent(data=f, media_type=media_type))

  # Build agent context from previous chat
  for user_msg, bot_msg in chat:
    if bot_msg is None:
      history.append(ModelMessage(role="user", content=user_msg))
    else:
      history.append(ModelMessage(role="assistant", content=bot_msg))

  async with agent.run_mcp_servers():
    response = await agent.run([message, *bin_files], message_history=history)
    output = response.output

  # Replace placeholder None with actual bot message
  chat[-1] = (message, output)
  save_history_to_disk(chat)

  # Clear in-memory history for next call
  history.clear()
  return chat, chat

# ========== Gradio Interface ==========
with gr.Blocks() as demo:
  chatbot = gr.Chatbot(value=initial_chat)
  msg = gr.Textbox(label="Message", placeholder="Type your message and press Enter")
  file_upload = gr.File(label="Upload files", file_types=[".txt", ".py", ".md", ".json"], file_count="multiple")
  chat_state = gr.State(initial_chat)

  with gr.Row():
    send_btn = gr.Button("Send")
    undo_btn = gr.Button("↩️ Undo")
    reset_btn = gr.Button("🔄 Reset")

  # Send button logic
  def sync_respond(message: str, files: list[bytes], chat: list[tuple[str, str]]):
    return asyncio.run(respond(message, files, chat))

  _ = send_btn.click(
    fn=sync_respond,
    inputs=[msg, file_upload, chat_state],
    outputs=[chatbot, chat_state]
  ).then(
    fn=lambda: "",
    inputs=None,
    outputs=msg
  )

  # Undo button logic
  def undo(chat: list[tuple[str, str]]):
    if chat:
      chat = chat[:-1]
      save_history_to_disk(chat)
    return chat, chat

  _ = undo_btn.click(
    fn=undo,
    inputs=[chat_state],
    outputs=[chatbot, chat_state]
  )

  # Reset button logic
  def reset_all() -> tuple[list[str], list[str], None]:
    # Clear disk and in-memory
    save_history_to_disk([])
    history.clear()
    return [], [], None

  _ = reset_btn.click(
    fn=reset_all,
    inputs=[],
    outputs=[chatbot, chat_state, file_upload]
  )

if __name__ == "__main__":
  _ = demo.launch()

