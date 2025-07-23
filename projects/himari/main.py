import modal
from pydantic import BaseModel

from evaluate import setup, evaluate

image = modal.Image.debian_slim().uv_sync()
app = modal.App(image=image, name="himari-evaluate")

class EvaluateRequest(BaseModel):
  author_name: str
  comment: str

@app.function()
@modal.fastapi_endpoint(method="POST", path="/evaluate")
def evaluate_endpoint(item: EvaluateRequest):
  model, tokenizer = setup()
  return {"prob": evaluate(model, tokenizer, item.author_name, item.comment)}

