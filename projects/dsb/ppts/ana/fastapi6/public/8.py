from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
  id: str
  password: str
  comment: str | None = None

@app.get("/", response_model_exclude_unset=True)
async def test() -> User:
  return User(id="", password="")

