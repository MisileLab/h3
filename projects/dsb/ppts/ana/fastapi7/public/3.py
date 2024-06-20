from fastapi import FastAPI, status
from pydantic import BaseModel, EmailStr

app = FastAPI()

class BaseUser(BaseModel):
  username: str
  email: EmailStr
  full_name: str | None = None

class User(BaseUser):
  password: str

@app.post("/user", status_code=status.HTTP_202_ACCEPTED)
async def create_user(user: User) -> BaseUser:
  print(user.model_dump())
  print(user.model_dump_json())
  return user