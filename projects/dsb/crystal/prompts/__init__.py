from os import getenv
from pathlib import Path

prompts = {
  v.stem: v.read_text().replace("<Username />", getenv("USER_ID", "misile"))
  for v in Path("./prompts").glob("*")
  if v.is_file() and v.name != "__init__.py"
}

