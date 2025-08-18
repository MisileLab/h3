from dataclasses import dataclass
from enum import Enum

class Stage(Enum):
  PREPARE = "prepare"
  EXPLORE = "explore"
  RESULT = "result"

