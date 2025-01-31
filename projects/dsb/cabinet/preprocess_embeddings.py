import torch
from torch import save, tensor, cat
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm
from pandas import read_json # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from pathlib import Path
from sys import platform

if not torch.cuda.is_available() and Path("zluda").exists() and platform == "win32":
  # rocm thing
  # https://github.com/lshqqytiger/ZLUDA
  torch.backends.cudnn.enabled = False
  torch.backends.cuda.enable_flash_sdp(False)
  torch.backends.cuda.enable_math_sdp(True)
  torch.backends.cuda.enable_mem_efficient_sdp(False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

amount = 3

x_t = tensor([], device=device)
y_t = tensor([], device=device)

for i in tqdm(list(Path("embedding_results").glob("*.jsonl"))):
  df = read_json(i, lines=True)
  df.sort_values(by="custom_id", key=lambda x: x.str.removeprefix("request-"), inplace=True) # pyright: ignore[reportUnknownLambdaType, reportUnknownMemberType]
  embedding: list[list[float]] = [x["body"]["data"][0]["embedding"] for x in df["response"]] # pyright: ignore[reportUnknownVariableType, reportUnknownVariableType]
  n = amount-1
  suicidal = i.name.startswith("suicidal")
  if len(embedding) < amount:
    continue
  t = tqdm(leave=False, total=len(embedding)-amount)
  while n < len(embedding):
    if len(embedding[n-amount:n]) < amount:
      n += 1
      _ = t.update(1)
      continue
    x_t = cat((x_t, tensor([embedding[n-amount:n]], device=device)))
    y_t = cat((y_t, tensor([1 if suicidal else 0], device=device)))
    n += 1
    _ = t.update(1)

dataset = TensorDataset(
  x_t,
  y_t
)
_ = save(dataset, "dataset.pt")
