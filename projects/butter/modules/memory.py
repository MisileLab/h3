from modules.llm_function import print_it
from .config import api_key, config

from pydantic import BaseModel, Field
from mem0 import Memory
from loguru import logger

from os import environ
from typing import Callable

mem0_config = {
  "vector_store": {
    "provider": "qdrant",
    "config": {
      "host": config['qdrant']['host'],
      "port": config['qdrant']['port']
    }
  },
  "llm": {
    "provider": "openai",
    "config": {
      "model": "gpt-4o-mini",
      "temperature": 0.2,
      "max_tokens": 1500
    }
  }
}

def preload(callback: Callable):
  environ['OPENAI_API_KEY'] = api_key
  res = callback()
  environ['OPENAI_API_KEY'] = ''
  return res

m: Memory = preload(lambda: Memory.from_config(config))

@print_it
def non_async_update_memory(id: str, content: str):
  return preload(lambda: m.update(id, content))

@print_it
def non_async_save_memory(username: str, content: str):
  return preload(lambda: m.add(content, username))

@print_it
def non_async_get_all_memories():
  return preload(lambda: m.get_all())

@print_it
def non_async_delete_memory(id: str):
  return preload(lambda: m.delete(id))

@print_it
async def save_memory(content: str, username: str | None = None):
  logger.debug(preload(lambda: m.add(content, username)))
  return 'success'

@print_it
async def get_all_memories(username: str | None = None):
  return str(preload(lambda: m.get_all(username)))

@print_it
async def get_memory(id: str):
  return str(preload(lambda: m.get(id)))

@print_it
async def search_memory(query: str, username: str | None = None):
  return str(preload(lambda: m.search(query, username, limit=10)))

@print_it
async def update_memory(id: str, content: str):
  logger.debug(preload(lambda: m.update(id, content)))
  return 'success'

@print_it
async def get_memory_history(id: str):
  return str(preload(lambda: m.history(id)))

@print_it
async def delete_memory(id: str):
  logger.debug(preload(lambda: m.delete(id)))
  return 'success'

class saveMemoryBase(BaseModel):
  """Save memory"""
  content: str = Field(..., content="The content of the memory")
  username: str | None = Field(None, content="The most related user's username")

class getAllMemoryBase(BaseModel):
  """Get all memories (limit is 100)"""
  username: str | None = Field(None, content="The most related user's username")

class getMemoryBase(BaseModel):
  """Get memory"""
  id: str = Field(..., content="The id of the memory")

class searchMemoryBase(BaseModel):
  """Search memory (limit is 10)"""
  query: str = Field(..., content="The query of the memory")
  username: str | None = Field(None, content="The most related user's username")

class updateMemoryBase(BaseModel):
  """Update memory that has specific id to content"""
  id: str = Field(..., content="The id of the memory")
  content: str = Field(..., content="The content of the memory")

class getMemoryHistoryBase(BaseModel):
  """Get memory history with id"""
  id: str = Field(..., content="The id of the memory")

class deleteMemoryBase(BaseModel):
  """Delete memory with id"""
  id: str = Field(..., content="The id of the memory")

functions = {
  "saveMemoryBase": saveMemoryBase,
  "getAllMemoryBase": getAllMemoryBase,
  "getMemoryBase": getMemoryBase,
  "searchMemoryBase": searchMemoryBase,
  "updateMemoryBase": updateMemoryBase,
  "getMemoryHistoryBase": getMemoryHistoryBase,
  "deleteMemoryBase": deleteMemoryBase
}

middle_converting_functions = {
  saveMemoryBase: save_memory,
  getAllMemoryBase: get_all_memories,
  getMemoryBase: get_memory,
  searchMemoryBase: search_memory,
  updateMemoryBase: update_memory,
  getMemoryHistoryBase: get_memory_history,
  deleteMemoryBase: delete_memory
}
