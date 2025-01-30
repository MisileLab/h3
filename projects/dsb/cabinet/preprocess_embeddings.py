import torch
from torch import Tensor, cat
from torch.utils.data import TensorDataset, DataLoader
from pandas import read_json # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]

from pathlib import Path
from os import getenv
from pickle import dumps

if not torch.cuda.is_available():
  # rocm thing
  # https://github.com/lshqqytiger/ZLUDA
  torch.backends.cudnn.enabled = False
  torch.backends.cuda.enable_flash_sdp(False)
  torch.backends.cuda.enable_math_sdp(True)
  torch.backends.cuda.enable_mem_efficient_sdp(False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

amount = 3

if getenv("COLAB_RELEASE_TAG"):
  from tqdm.notebook import tqdm
else:
  from tqdm import tqdm

x_t = Tensor(device=device)
y_t = Tensor(device=device)

for i in tqdm(list(Path("embedding_results").glob("*.jsonl"))):
  df = read_json(i, lines=True)
  df.sort_values(by="custom_id", key=lambda x: x.str.removeprefix("request-"), inplace=True) # pyright: ignore[reportUnknownLambdaType, reportUnknownMemberType]
  embedding: list[list[float]] = [x["body"]["data"][0]["embedding"] for x in df["response"]] # pyright: ignore[reportUnknownVariableType, reportUnknownVariableType]
  n = amount-1
  suicidal = i.name.startswith("suicidal")
  t = tqdm(leave=False, total=len(embedding)-amount)
  while n < len(embedding):
    if len(embedding[n-amount:n]) < amount:
      n += 1
      _ = t.update(1) # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
      continue
    x_t = cat((x_t, Tensor([embedding[n-amount:n]]))).to(device)
    y_t = cat((y_t, Tensor([1 if suicidal else 0]))).to(device)
    n += 1
    _ = t.update(1) # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

dataset = TensorDataset(
  x_t,
  y_t
)
batch_size = 32
data_loaders = DataLoader(dataset, batch_size=batch_size, shuffle=True)
_ = Path("model_data.pkl").write_bytes(dumps(data_loaders))
