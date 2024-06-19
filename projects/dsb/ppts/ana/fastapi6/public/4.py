from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
  id: str
  password: str

@app.post("/")
async def test(user: User) -> User:
  if user.password != "password":
    return ""
  return user

