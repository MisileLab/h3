from fastapi import FastAPI
from pydantic import BaseModel, EmailStr

app = FastAPI()

class BaseUser(BaseModel):
  username: str
  email: EmailStr
  full_name: str | None = None

@app.post("/user")
async def create_user(user: BaseUser) -> BaseUser:
  print(user.model_dump())
  print(user.model_dump_json())
  return user