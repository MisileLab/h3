from dataclasses import dataclass
from pathlib import Path
from json import loads
from typing import final, override

from tqdm import tqdm
from torch import Tensor, nn, cat, ones, optim, zeros, save # pyright: ignore[reportUnknownVariableType]
from torch.utils.data import TensorDataset, DataLoader

@final
class Model(nn.Module):
  def __init__(self, data_amount: int) -> None:
    super().__init__() # pyright: ignore[reportUnknownMemberType]
    self.embedding_layers: list[nn.Sequential] = [nn.Sequential(
      nn.Linear(4096, 64),
      nn.ReLU(),
      nn.Dropout(0.2)
    ) for _ in range(data_amount)]

    self.percentage_layer = nn.Sequential()
    n = 64 * data_amount
    while n // 4 >= 32:
      _ = self.percentage_layer.append(nn.Linear(n, n // 4))
      _ = self.percentage_layer.append(nn.ReLU())
      _ = self.percentage_layer.append(nn.Dropout(0.2))
    _ = self.percentage_layer.append(nn.Linear(32, 1))

  @override
  def forward(self, x: list[Tensor]) -> Tensor:
    embedding_layer_results: list[Tensor] = [
      f(i) for f, i in zip(self.embedding_layers, x)
    ]

    return self.percentage_layer(cat(embedding_layer_results, dim=1)) # pyright: ignore[reportAny]

amount = 3

@final
@dataclass
class Data:
  normal = Tensor()
  suicidal = Tensor()

result = Data()

for i in tqdm(Path("embedding_results").glob("*.jsonl")):
  data = sorted(loads(i.read_text()), key=lambda x: int(x["custom-id"].removeprefix("request-"))) # pyright: ignore[reportAny]
  embedding: list[list[float]] = [x["response"]["body"]["data"][0]["embedding"] for x in data] # pyright: ignore[reportAny]
  n = amount-1
  suicidal = i.name.startswith("suicidal")
  t = tqdm(leave=False, total=len(embedding)-amount)
  while n < len(embedding):
    if suicidal:
      result.suicidal = cat([result.suicidal, Tensor(embedding[n-amount:n+1])], dim=1)
    else:
      result.normal = cat([result.normal, Tensor(embedding[n-amount:n+1])], dim=1)
    n += 1
    _ = t.update(1)

dataset = TensorDataset(
  cat([result.normal, result.suicidal], dim=1),
  cat([zeros(result.normal.shape[1]), ones(result.suicidal.shape[1])])
)
batch_size = 32
data_loaders = DataLoader(dataset, batch_size=batch_size, shuffle=True)

epoches = 300
model = Model(amount)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

model = Model(amount)
for _ in tqdm(range(epoches)):
  _ = model.train()
  for x, y in data_loaders: # pyright: ignore[reportAny]
    forward: Tensor = model(x)
    loss = criterion(forward, y) # pyright: ignore[reportAny]

    optimizer.zero_grad()
    loss.backward() # pyright: ignore[reportAny]
    _ = optimizer.step() # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

save(model.state_dict(), "model.pth")

