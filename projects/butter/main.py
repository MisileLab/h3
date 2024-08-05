from modules.llm_function import middle_prompt, llm_mini
from modules.llm import api_key, llm, functions, middle_converting_functions
from modules.memory import print_it, non_async_save_memory, non_async_get_all_memories, non_async_update_memory, non_async_delete_memory
from modules.config import config
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

import gradio as gr
from loguru import logger
from PIL import Image
from openai import OpenAI
from binaryornot.check import is_binary

from pathlib import Path
from base64 import b64encode
from datetime import datetime
from copy import deepcopy
from typing import Any

try:
  user # type: ignore
except NameError:
  # first launch before gradio reload
  user = ""
  prompt = Path("./prompts/prompt").read_text()
  summarize_prompt = Path("./prompts/summarize_prompt").read_text()
  temp = {
    "prompt": prompt,
    "middle_prompt": middle_prompt,
    "summarize_prompt": summarize_prompt,
    "memory_id": "",
    "memory_content": "",
    "memory_user": "",
    "user": "",
    "voice": "",
    "files": []
  }
  messages: list[SystemMessage | AIMessage | ToolMessage | HumanMessage] = [SystemMessage(prompt)]
  whisper = OpenAI(api_key=api_key)

def persist_temp(key: str, value: Any):
  temp[key] = value
  return temp[key]

@print_it
def process_memories(memories: list[dict]):
  a = []
  for i in memories:
    tmp = i.get("created_at", None)
    if tmp is None:
      tmp = i.get("updated_at", 0)
    a.append([i["id"], datetime.fromtimestamp(tmp), i.get("user_id", ""), i["text"]])
  return a

def convert_to_real_content(content_text, content_files, voice, user):
  result = ""
  logger.debug(voice)
  result_images = []
  for i in content_files:
    logger.info(i)
    if not is_binary(i):
      logger.info("its text file")
      result += f"{Path(i).name}:\n```{Path(i).read_text()}```"
    else:
      logger.info("File is not a text file, we treat as image")
      try:
        Image.open(i).verify()
        file_type = ""
        if i.endswith(".png"):
          file_type = "png"
        elif i.endswith(".jpg") or i.endswith(".jpeg"):
          file_type = "jpeg"
        elif i.endswith(".webp"):
          file_type = "webp"
        else:
          raise gr.Error("File is not supported")
        # file deepcode ignore HandleUnicode: idk what is this
        result_images.append(f"data:image/{file_type};base64,{b64encode(Path(i).read_bytes()).decode('utf-8')}")
      except IOError as exc:
        raise gr.Error("File is not an image file") from exc
  if voice is not None and voice != "":
    logger.info(voice)
    result += f"\naudio: {whisper.audio.transcriptions.create(file=open(voice, "rb"), model="whisper-1").text}"
  if content_text != "":
    result += f"\n{content_text}"
  logger.debug(len(content_files))
  return (f"message author's name: {user}\n{result}", result_images)

def reset():
  logger.info("resetted")
  global messages
  messages = [SystemMessage(prompt)]

async def generate_message(content, _):
  if user is None or user == "":
    raise gr.Error("no user???")
  if temp["files"] is None:
    temp["files"] = []
  logger.debug(content)
  logger.info(content, temp["files"])
  msg = convert_to_real_content(content, temp["files"], temp["voice"], user)
  global messages
  messages.append(HumanMessage([{"type": "text", "text": msg[0]}] + [{"type": "image_url", "image_url": {"url": i}} for i in msg[1]])) # type: ignore
  while True:
    first = True
    logger.debug(messages)
    async for i in llm.astream(messages):
      if first:
        gathered = i
        first = False
      else:
        gathered += i # type: ignore gathered not unbound
      logger.debug(i)
      yield gathered.content # type: ignore gathered is ai's message
    messages.append(gathered) # type: ignore gathered not unbound
    if len(gathered.tool_calls) == 0: # type: ignore gathered is ai's message
      break
    for i in gathered.tool_calls: # type: ignore gathered is ai's message
      logger.info(f"calling {i["name"]}")
      messages.append(ToolMessage(tool_call_id=i["id"], content=await middle_converting_functions[functions[i["name"]]](**i["args"])))
  if len(messages) >= 70:
    tmp_messages = messages[1:60]
    while tmp_messages[-1].__class__ in [ToolMessage, HumanMessage]:
      logger.debug(f"stripping {tmp_messages[-1]}, because it doesn't ending with ai message")
      tmp_messages = tmp_messages[:-1]
    summarized = llm_mini.invoke([SystemMessage(summarize_prompt)] + deepcopy(tmp_messages)).content
    tmp_messages = messages[60:]
    while tmp_messages[0].__class__ == ToolMessage:
      logger.debug(f"stripping {tmp_messages[0]}, because it's starting with tool message")
      tmp_messages = tmp_messages[1:]
    messages = [SystemMessage(prompt), HumanMessage(summarized), AIMessage("알았어!")] + deepcopy(tmp_messages)

@print_it
def confirm():
  global prompt
  global middle_prompt
  global summarize_prompt
  prompt = temp['prompt']
  messages[0] = SystemMessage(prompt)
  middle_prompt = temp['middle_prompt']
  summarize_prompt = temp['summarize_prompt']
  Path("./prompts/prompt").write_text(temp['prompt'])
  Path("./prompts/middle_prompt").write_text(temp['middle_prompt'])
  global user
  user = temp["user"]
  return [user, prompt, middle_prompt, summarize_prompt]

with gr.Blocks() as frontend:
  with gr.Tab("Chatting"):
    chat = gr.ChatInterface(generate_message)
    with gr.Accordion("more inputs", open=False):
      voice = gr.Audio(type="filepath", label="Audio")
      voice.change(lambda x: temp.__setitem__("voice", x), voice)

      file = gr.File(type="filepath", label="Files", file_count="multiple")
      file.change(lambda x: temp.__setitem__("files", x), file)
  with gr.Tab("Configuration"):
    user_input = gr.Textbox(label="user")
    user_input.change(lambda x: temp.__setitem__("user", x), user_input)

    prompt_input = gr.Textbox(label="prompt", show_copy_button=True, interactive=True, lines=13) # type: ignore not unbound
    prompt_input.change(lambda x: temp.__setitem__("prompt", x), prompt_input)

    middle_prompt_input = gr.Textbox(label="middle_prompt", show_copy_button=True, interactive=True)
    middle_prompt_input.change(lambda x: temp.__setitem__("middle_prompt", x), middle_prompt_input)

    summarize_prompt_input = gr.Textbox(label="summarize_prompt", show_copy_button=True, interactive=True) # type: ignore not unbound
    summarize_prompt_input.change(lambda x: temp.__setitem__("summarize_prompt", x), summarize_prompt_input)

    confirm_button = gr.Button("Confirm")
    confirm_button.click(confirm, None, [user_input, prompt_input, middle_prompt_input, summarize_prompt_input])
  with gr.Tab("Admin Panel"):
    reset_history = gr.Button("Reset History", variant="stop")
    reset_history.click(reset)
    
    df = gr.Dataframe(label="memories", headers=["id", "created_at", "user", "text"], datatype=["str", "date", "str", "str"], value=process_memories(non_async_get_all_memories()))
    refresh = gr.Button("Refresh", variant="secondary")
    refresh.click(lambda: process_memories(non_async_get_all_memories()), None, df)
    
    memory_id = gr.Textbox(label="id of memory that modify", value=temp["memory_id"]) # type: ignore temp not unbound
    memory_id.change(lambda x: temp.__setitem__("memory_id", x), memory_id)
    
    memory_content = gr.Textbox(label="content of memory that modify", value=temp["memory_content"]) # type: ignore temp not unbound
    memory_content.change(lambda x: temp.__setitem__("memory_content", x), memory_content)
    
    memory_user = gr.Textbox(label="user of memory that modify", value=temp["memory_user"]) # type: ignore temp not unbound
    memory_user.change(lambda x: temp.__setitem__("memory_user", x), memory_user)
    
    memory_save = gr.Button("Save", variant="primary")
    memory_save.click(lambda: non_async_save_memory(temp["memory_user"], temp["memory_content"]))
    
    memory_update = gr.Button("Update", variant="primary")
    memory_update.click(lambda: non_async_update_memory(temp["memory_id"], temp["memory_content"]))
    
    memory_delete = gr.Button("Delete", variant="stop")
    memory_delete.click(lambda: non_async_delete_memory(temp["memory_id"]))
  frontend.load(lambda: [user, prompt, middle_prompt, summarize_prompt], None, [user_input, prompt_input, middle_prompt_input, summarize_prompt_input])

frontend.launch(show_error=True, show_api=True, auth=(config['auth']['id'], config['auth']['password']), server_name='0.0.0.0')
