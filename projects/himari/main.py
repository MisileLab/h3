import modal
from fastapi import Request, HTTPException, status
from pydantic import BaseModel

from evaluate import setup, evaluate

image = modal.Image.debian_slim().uv_sync()
app = modal.App(image=image, name="himari-evaluate")

class EvaluateRequest(BaseModel):
  author_name: str
  comment: str

@app.function()
@modal.fastapi_endpoint(method="POST", docs=True) # pyright: ignore[reportUnknownMemberType]
def evaluate_endpoint(request: Request, item: EvaluateRequest):
  if not request.client or request.client.host != "211.219.106.17":
    raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Forbidden")
  model, tokenizer = setup() # pyright: ignore[reportUnknownVariableType]
  return {"prob": evaluate(model, tokenizer, item.author_name, item.comment)} # pyright: ignore[reportUnknownArgumentType]

