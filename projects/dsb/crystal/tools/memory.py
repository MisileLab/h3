from os import getenv
from typing import Callable

from mem0 import Memory # pyright: ignore[reportMissingTypeStubs]

m = Memory.from_config({
  "llm": {
    "config": {"model": "gpt-4.1"}
  },
  "graph_store": {
    "provider": "memgraph",
    "config": {
      "url": getenv("MEMGRAPH_URL"),
      "username": getenv("MEMGRAPH_USERNAME"),
      "password": getenv("MEMGRAPH_PASSWORD"),
    },
  },
  "vector_store": {
    "provider": "qdrant",
    "config": {
      "collection_name": "crystal",
      "host": getenv("QDRANT_HOST"),
      "port": getenv("QDRANT_PORT"),
      "api_key": getenv("QDRANT_API_KEY"),
      "embedding_model_dims": 3072
    }
  },
  "embedder": {
    "provider": "openai",
    "config": {
      "model": "text-embedding-3-large",
      "embedding_dims": 3072
    }
  }
})

user_id = getenv("USER_ID", "misile")

def add_memory(query: str): # pyright: ignore[reportUnknownParameterType]
  return m.add(query, user_id=user_id) # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

def get_all_memory(): # pyright: ignore[reportUnknownParameterType]
  return m.get_all(user_id=user_id) # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

def search_memory(query: str): # pyright: ignore[reportUnknownParameterType]
  return m.search(query, user_id=user_id) # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

tools: list[Callable] = [add_memory, get_all_memory, search_memory] # pyright: ignore[reportMissingTypeArgument, reportUnknownVariableType]

