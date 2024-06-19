from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class User(BaseModel):
  id: str
  password: str

class BaseUser(BaseModel):
  id: str

@app.post("/")
async def test(user: User) -> BaseUser:
  if user.password != "password":
    return BaseUser(id="asdf")
  return user

