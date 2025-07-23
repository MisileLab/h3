from argon2 import PasswordHasher
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from os import getenv

app = FastAPI()
ph = PasswordHasher()
api_key = getenv("API_KEY")
if not api_key:
  raise ValueError("API_KEY environment variable is not set")
api_key_hashed = PasswordHasher().hash(api_key)

class CollectRequest(BaseModel):
  author_name: str
  content: str
  is_bot: bool
  api_key: str

@app.post("/collect")
async def collect_data(item: CollectRequest):
  if not ph.verify(api_key_hashed, item.api_key):
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
  if not item.author_name or not item.content:
    raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Author name and content are required")
  # TODO: make actual database call

