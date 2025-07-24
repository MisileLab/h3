import modal
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from os import getenv
import uuid
from datetime import datetime

from evaluator import Model

_ = load_dotenv()

# Modal setup
app = modal.App(name="himari-api")
web_app = FastAPI()
ph = PasswordHasher()
model = None

# Create Modal image with dependencies
image = modal.Image.debian_slim().uv_sync()
image = image.add_local_file("model.safetensors", "/models/model.safetensors")
image = image.add_local_python_source("evaluator")
image = image.add_local_python_source("evaluate")

# Create KV store
kv_store = modal.Dict.from_name("himari-data", create_if_missing=True)

# Define the Modal web endpoint
@app.function(image=image, secrets=[modal.Secret.from_name("API_KEY")], gpu="any")
@modal.asgi_app()
def fastapi_app():
  return web_app

class ReportRequest(BaseModel):
  author_name: str
  content: str
  is_bot: bool
  api_key: str

class EvaluateRequest(BaseModel):
  author_name: str
  content: str
  api_key: str

@web_app.post("/evaluate")
async def evaluate(item: EvaluateRequest) -> dict[str, float]:
  api_key = getenv("API_KEY")
  if not api_key:
    raise ValueError("API_KEY environment variable is not set")
  api_key_hashed = PasswordHasher().hash(api_key)
  try:
    _ = ph.verify(api_key_hashed, item.api_key)
  except VerifyMismatchError as e:
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid API key") from e
  if not item.author_name or not item.content:
    raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Author name and content are required")
  
  global model
  # Evaluate using Modal remote execution
  if model is None:
    model = Model()
    model.setup()
  result: float = model.evaluate(item.author_name, item.content)
  is_bot = result > 0.9
  
  # Store in Modal's KV store
  entry_id = str(uuid.uuid4())
  kv_store[entry_id] = {
    "author_name": item.author_name,
    "content": item.content,
    "is_bot": is_bot,
    "timestamp": datetime.now().isoformat()
  }
  
  return {"result": result}

@web_app.post("/report")
async def report(item: ReportRequest):
  api_key = getenv("API_KEY")
  if not api_key:
    raise ValueError("API_KEY environment variable is not set")
  api_key_hashed = PasswordHasher().hash(api_key)
  try:
    _ = ph.verify(api_key_hashed, item.api_key)
  except VerifyMismatchError as e:
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid API key") from e
  if not item.author_name or not item.content:
    raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Author name and content are required")
  
  # Store in Modal's KV store
  entry_id = str(uuid.uuid4())
  kv_store[entry_id] = {
    "author_name": item.author_name,
    "content": item.content,
    "is_bot": item.is_bot,
    "timestamp": datetime.now().isoformat()
  }
  
  return {"result": "success"}

# Local entrypoint for development
@app.local_entrypoint()
def main():
  print("Starting Himari API server...")
  modal.run("himari-api:fastapi_app")
