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
from httpx import AsyncClient

from prompts import prompts
from tools.feedback import (
  tools as feedback_tools,  # pyright: ignore[reportUnknownVariableType]
)
from tools.memory import (
  tools as memory_tools,  # pyright: ignore[reportUnknownVariableType]
)

github_mcp_server = MCPServerStdio(
  'docker',
  args=[
    "run",
    "-i",
    "--rm",
    "-e",
    "GITHUB_PERSONAL_ACCESS_TOKEN",
    "ghcr.io/github/github-mcp-server"
  ],
  env={
    "GITHUB_PERSONAL_ACCESS_TOKEN": getenv("GITHUB_API_KEY", "")
  },
  timeout=20
)

google_calendar_mcp = MCPServerStdio(
  'pnpx',
  args=['@cocal/google-calendar-mcp'],
  env={
    "GOOGLE_OAUTH_CREDENTIALS": "./credentials.json"
  },
  timeout=20
)

arxiv_mcp_server = MCPServerStdio(
  'direnv',
  args=[
    "exec",
    ".",
    "uv",
    "tool",
    "run",
    "arxiv-mcp-server",
    "--storage-path",
    "./cache"
  ],
  timeout=20
)

mcp_servers = [
  google_calendar_mcp,
  arxiv_mcp_server,
  github_mcp_server
]

# ========== Setup ==========
model = OpenAIModel(
  'google/gemini-2.5-flash',
  provider=OpenAIProvider(
    base_url='https://openrouter.ai/api/v1',
    api_key=getenv('OPENROUTER_KEY')
  )
)

summarize_model = OpenAIModel(
  'gpt-5-nano',
  provider=OpenAIProvider(api_key=getenv("OPENAI_KEY"))
)

tools = [ # pyright: ignore[reportUnknownVariableType]
  duckduckgo_search_tool(),
  *memory_tools,
  *feedback_tools
]

agent = Agent(
  model,
  tools=tools, # pyright: ignore[reportUnknownArgumentType]
  mcp_servers=mcp_servers,
  instructions=prompts["main"]
)

web_agent = Agent(
  summarize_model,
  instructions=prompts["web"]
)

_ = configure(token=getenv('LOGFIRE_KEY'))
_ = instrument_openai()

@agent.tool_plain
async def get_page(url: str) -> str:
  resp = ""
  async with AsyncClient(timeout=None) as client:
    async with client.stream("POST", "https://r.jina.ai", headers={
      "Authorization": f"Bearer {getenv('JINA_API_KEY')}",
      "X-Engine": "Browser",
      "Accept": "text/event-stream"
      }, data={
        "url": url
      }, timeout=None) as response:
        async for line in response.aiter_lines():
          if line.startswith("data:"):
            data_line = line.removeprefix("data:").strip()
            resp = data_line 
  return resp

if Path("data.pkl").exists():
  initial = loads(Path("data.pkl").read_bytes()) # pyright: ignore[reportAny]
else:
  initial: tuple[list[ModelMessage], list[tuple[str, str]]] = ([], [])

history = initial[0]
chat_state = initial[1]

# ========== Streaming Chat Function ==========
async def respond_stream(message: str, files: list[bytes]):
  """Streaming response generator with reasoning and tool call display"""
  bin_files: list[BinaryContent] = []
  if files:
    for f in files:
      media_type = from_stream(f, mime=True)
      bin_files.append(BinaryContent(data=f, media_type=media_type))

  # Add user message to chat state immediately
  current_chat = chat_state + [(message, "")]
  yield current_chat
  
  try:
    async with agent.run_mcp_servers():
      # Use stream method instead of run for streaming
      response_stream = agent.stream([message, *bin_files], message_history=history)
      
      accumulated_response = ""
      reasoning_steps = []
      tool_calls = []
      
      async for chunk in response_stream:
        display_text = ""
        
        # Handle different types of streaming chunks
        if hasattr(chunk, 'delta'):
          # This is likely a text delta
          if chunk.delta:
            accumulated_response += chunk.delta
        elif hasattr(chunk, 'output'):
          # Complete output update
          accumulated_response = chunk.output
        elif hasattr(chunk, 'content'):
          # Content chunk
          accumulated_response += chunk.content
        
        # Check for reasoning/thinking content
        if hasattr(chunk, 'reasoning') and chunk.reasoning:
          reasoning_steps.append(chunk.reasoning)
        elif hasattr(chunk, 'thought') and chunk.thought:
          reasoning_steps.append(chunk.thought)
        elif hasattr(chunk, 'thinking') and chunk.thinking:
          reasoning_steps.append(chunk.thinking)
        
        # Check for tool calls
        if hasattr(chunk, 'tool_calls') and chunk.tool_calls:
          for tool_call in chunk.tool_calls:
            tool_info = f"🔧 **{tool_call.get('function', {}).get('name', 'Unknown Tool')}**"
            if 'function' in tool_call and 'arguments' in tool_call['function']:
              args = tool_call['function']['arguments']
              if isinstance(args, str) and len(args) < 200:
                tool_info += f"\n```json\n{args}\n```"
              elif isinstance(args, dict):
                import json
                tool_info += f"\n```json\n{json.dumps(args, indent=2)[:200]}{'...' if len(json.dumps(args)) > 200 else ''}\n```"
            tool_calls.append(tool_info)
        elif hasattr(chunk, 'function_call') and chunk.function_call:
          # Handle OpenAI-style function calls
          func_call = chunk.function_call
          tool_info = f"🔧 **{func_call.get('name', 'Unknown Tool')}**"
          if 'arguments' in func_call:
            args = func_call['arguments']
            if len(args) < 200:
              tool_info += f"\n```json\n{args}\n```"
            else:
              tool_info += f"\n```json\n{args[:200]}...\n```"
          tool_calls.append(tool_info)
        
        # Check for tool results
        if hasattr(chunk, 'tool_result') and chunk.tool_result:
          result = chunk.tool_result
          tool_result_info = f"✅ **Tool Result**\n```\n{str(result)[:300]}{'...' if len(str(result)) > 300 else ''}\n```"
          tool_calls.append(tool_result_info)
        
        # Build the display text based on user preference
        if show_reasoning:
          if reasoning_steps:
            display_text += "\n\n**🤔 Reasoning:**\n"
            for i, step in enumerate(reasoning_steps[-3:]):  # Show last 3 reasoning steps
              display_text += f"{i+1}. {step}\n"
          
          if tool_calls:
            display_text += "\n\n**🛠️ Tool Usage:**\n"
            for tool_call in tool_calls[-5:]:  # Show last 5 tool calls
              display_text += f"{tool_call}\n\n"
          
          if accumulated_response:
            display_text += f"\n\n**💬 Response:**\n{accumulated_response}"
        else:
          # Only show the response when reasoning is disabled
          display_text = accumulated_response
        
        # Update the chat state with current progress
        current_chat = chat_state + [(message, display_text.strip())]
        yield current_chat
      
      # Final update with complete response
      final_response = None
      try:
        if hasattr(response_stream, 'finalize'):
          final_response = await response_stream.finalize()
        else:
          # If no finalize method, the last chunk should contain the final response
          final_response = chunk if 'chunk' in locals() else None
      except:
        # Fallback to regular run if streaming completion fails
        final_response = await agent.run([message, *bin_files], message_history=history)
      
      if final_response:
        history.clear()
        if hasattr(final_response, 'all_messages'):
          history.extend(final_response.all_messages())
        elif hasattr(final_response, 'messages'):
          history.extend(final_response.messages)
        
        # Clean final response (remove reasoning/tool info for history)
        clean_response = final_response.output if hasattr(final_response, 'output') else accumulated_response
        chat_state.append((message, clean_response))
      else:
        # Fallback: use accumulated response
        chat_state.append((message, accumulated_response))
      
      yield chat_state
      
  except Exception as e:
    # Enhanced fallback to non-streaming if streaming fails
    print(f"Streaming failed, falling back to regular response: {e}")
    try:
      async with agent.run_mcp_servers():
        response = await agent.run([message, *bin_files], message_history=history)
      history.clear()
      history.extend(response.all_messages())
      chat_state.append((message, response.output))
      yield chat_state
    except Exception as fallback_error:
      # Ultimate fallback
      error_message = f"❌ Error: {str(fallback_error)}"
      chat_state.append((message, error_message))
      yield chat_state

async def summarize():
  async with agent.run_mcp_servers():
    output = await agent.run(
      "summarize our conversation to optimize message history!",
      message_history=history
    )
  history.clear()
  history.extend(output.new_messages())
  chat_state.clear()
  chat_state.append(("", output.output))
  return chat_state

def save():
  """Save current state to pickle file"""
  Path("data.pkl").write_bytes(dumps([history, chat_state]))

# ========== Global Settings ==========
show_reasoning = True  # Global flag to control reasoning display

# ========== Gradio Interface ==========
with gr.Blocks(css="""
  .reasoning-section { 
    background-color: #f8f9fa; 
    padding: 10px; 
    border-left: 3px solid #007bff; 
    margin: 5px 0; 
  }
  .tool-section { 
    background-color: #fff3cd; 
    padding: 10px; 
    border-left: 3px solid #ffc107; 
    margin: 5px 0; 
  }
""") as demo:
  
  with gr.Row():
    with gr.Column(scale=4):
      chatbot = gr.Chatbot(
        height=500,
        show_copy_button=True,
        render_markdown=True
      )
    with gr.Column(scale=1):
      with gr.Group():
        gr.Markdown("### Settings")
        reasoning_toggle = gr.Checkbox(
          label="Show reasoning & tool calls",
          value=True,
          info="Display AI's thinking process and tool usage"
        )
        
  msg = gr.Textbox(
    label="Message", 
    placeholder="Type your message and press Enter",
    lines=2,
    max_lines=8
  )
  file_upload = gr.File(
    label="Upload files", 
    file_count="multiple", 
    file_types=["file"]
  )
  
  with gr.Row():
    send_btn = gr.Button("Send", variant="primary")
    undo_btn = gr.Button("↩️ Undo")
    reset_btn = gr.Button("🔄 Reset")
    summarize_btn = gr.Button("📝 Summarize")

  # Streaming send button logic
  def sync_respond_stream(message: str, files: list[bytes], show_reasoning_flag: bool):
    """Synchronous wrapper for streaming response"""
    global show_reasoning
    show_reasoning = show_reasoning_flag
    
    async def run_stream():
      async for result in respond_stream(message, files):
        yield result
    
    # Run the async generator
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
      async_gen = run_stream()
      while True:
        try:
          result = loop.run_until_complete(async_gen.__anext__())
          yield result
        except StopAsyncIteration:
          break
    finally:
      loop.close()

  # Send button with streaming
  send_event = send_btn.click(
    fn=sync_respond_stream,
    inputs=[msg, file_upload, reasoning_toggle],
    outputs=[chatbot]
  ).then(
    fn=lambda: ("", None),  # Clear message and file upload
    inputs=None,
    outputs=[msg, file_upload]
  ).then(
    fn=save,
    inputs=[],
    outputs=None
  )

  # Enter key support for sending messages
  msg.submit(
    fn=sync_respond_stream,
    inputs=[msg, file_upload, reasoning_toggle],
    outputs=[chatbot]
  ).then(
    fn=lambda: ("", None),
    inputs=None,
    outputs=[msg, file_upload]
  ).then(
    fn=save,
    inputs=[],
    outputs=None
  )

  # Undo button logic
  def undo():
    if len(chat_state) >= 1:
      if len(history) >= 2:
        history.pop()
        history.pop()
      chat_state.pop()
    return chat_state

  undo_btn.click(
    fn=undo,
    inputs=[],
    outputs=[chatbot]
  ).then(
    fn=save,
    inputs=[],
    outputs=None
  )

  # Reset button logic
  def reset_all():
    history.clear()
    chat_state.clear()
    return [], "", None

  reset_btn.click(
    fn=reset_all,
    inputs=[],
    outputs=[chatbot, msg, file_upload]
  ).then(
    fn=save,
    inputs=[],
    outputs=None
  )

  # Summarize button with async wrapper
  def sync_summarize():
    return asyncio.run(summarize())

  summarize_btn.click(
    fn=sync_summarize,
    inputs=[],
    outputs=[chatbot]
  ).then(
    fn=save,
    inputs=[],
    outputs=None
  )

  # Load chat state on startup
  demo.load(fn=lambda: chat_state, outputs=[chatbot])

async def main():
  demo.launch()

if __name__ == "__main__":
  asyncio.run(main())
