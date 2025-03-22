from enum import Enum

from pydantic import BaseModel

class Role(Enum):
  system = "system"
  user = "user"
  assistant = "assistant"

class Data(BaseModel):
  conversations: list[str]
  analysis: list[str]
  suicidal: bool

class Message(BaseModel):
  role: Role
  content: str

