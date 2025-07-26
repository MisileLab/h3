import modal
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from blake3 import blake3
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from collections.abc import Sequence
from os import getenv
from datetime import datetime

from evaluator import Model

_ = load_dotenv()

# Modal setup
app = modal.App(name="himari-api")
web_app = FastAPI()

# Add CORS middleware
web_app.add_middleware(
  CORSMiddleware,
  allow_origins=["chrome-extension://pjahmjjajanjbkkekkiihmdkchkajjdp"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

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

class EvaluateRequestSchema(BaseModel):
  author_name: str
  content: str

class EvaluateRequest(BaseModel):
  evaluate: list[EvaluateRequestSchema]
  api_key: str

@web_app.post("/evaluate")
async def evaluate(item: EvaluateRequest) -> dict[str, Sequence[float | bool]]:
  if not item.evaluate:
    raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Author name and content are required")
  if len(item.evaluate) >= 50:
    raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Too many fields in request")
  api_key = getenv("API_KEY")
  if not api_key:
    raise ValueError("API_KEY environment variable is not set")
  api_key_hashed = PasswordHasher().hash(api_key)
  try:
    _ = ph.verify(api_key_hashed, item.api_key)
  except VerifyMismatchError as e:
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid API key") from e

  global model
  # Evaluate using Modal remote execution
  if model is None:
    model = Model()
    model.setup()
  author_names: list[str] = []
  contents: list[str] = []
  result: list[float] = []
  for i in item.evaluate:
    if kv_store.get(blake3((i.author_name + i.content).encode()).digest()):
      result.append(kv_store.get(blake3((i.author_name + i.content).encode()).digest())["result"])
    else:
      author_names.append(i.author_name)
      contents.append(i.content)
  if author_names:
    result.extend(model.evaluate(author_names, contents))
  is_bot = [result > 0.9 for result in result]

  # Store in Modal's KV store
  for i in zip(author_names, contents, result, is_bot):
    if kv_store.get(blake3((i[0] + i[1]).encode()).digest()):
      continue
    kv_store[blake3((i[0] + i[1]).encode()).digest()] = {
      "author_name": i[0],
      "content": i[1],
      "is_bot": i[3],
      "result": i[2],
      "timestamp": datetime.now().isoformat()
    }

  return {"result": result, "is_bot": is_bot}

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
  kv_store[blake3((item.author_name + item.content).encode()).digest()] = {
    "author_name": item.author_name,
    "content": item.content,
    "is_bot": item.is_bot,
    "result": 1 if item.is_bot else 0,
    "timestamp": datetime.now().isoformat()
  }
  
  return {"result": "success"}

# Local entrypoint for development
@app.local_entrypoint()
def main():
  print("Starting Himari API server...")
  modal.run("himari-api:fastapi_app")
