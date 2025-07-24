import modal

from pathlib import Path
from typing import final

image = modal.Image.debian_slim().uv_pip_install("torch", "transformers", "safetensors")
volume = modal.Volume.from_name("himari-models", create_if_missing=True) # pyright: ignore[reportUnknownMemberType]
MODEL_DIR = Path("/models")

app = modal.App(image=image, name="himari-evaluate")

@final
class Model:
  def setup(self):
    from evaluate import setup as model_setup
    model, tokenizer = model_setup() # pyright: ignore[reportUnknownVariableType]
    self.model = model # pyright: ignore[reportUninitializedInstanceVariable]
    self.tokenizer = tokenizer # pyright: ignore[reportUninitializedInstanceVariable, reportUnknownMemberType]
  
  def evaluate(self, author_name: str, comment: str):
    from evaluate import evaluate as model_evaluate
    return model_evaluate(self.model, self.tokenizer, author_name, comment) # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
