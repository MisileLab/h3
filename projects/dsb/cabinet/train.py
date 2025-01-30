from typing import final
from sys import version_info
from pathlib import Path
from pickle import loads

if version_info < (3, 13):
  from typing_extensions import override
else:
  from typing import override

from tqdm.auto import tqdm

from torch import Tensor, nn, cat, optim, save
from torch.cuda import is_available
from torch.utils.data import DataLoader

device = "cuda" if is_available() else "cpu"
print(device)

@final
class Model(nn.Module):
  def __init__(self, data_amount: int) -> None:
    super().__init__()  # pyright: ignore[reportUnknownMemberType]
    self.embedding_layers = nn.ModuleList([
      nn.Sequential(
        nn.Linear(3072, 64),
        nn.ReLU(),
        nn.Dropout(0.2)
      ) for _ in range(data_amount)
    ])

    self.percentage_layer = nn.Sequential()
    n = 64 * data_amount
    while n // 4 >= 32:
      _ = self.percentage_layer.append(nn.Linear(n, n // 4))
      _ = self.percentage_layer.append(nn.ReLU())
      _ = self.percentage_layer.append(nn.Dropout(0.2))
      n //= 4
    _ = self.percentage_layer.append(nn.Linear(n, 1))
    self.percentage_layer = self.percentage_layer

  @override
  def forward(self, x: Tensor) -> Tensor:
    embedding_layer_results: list[Tensor] = []
    for i, f in enumerate(self.embedding_layers):
      xi = x[:, i, :]
      _ = embedding_layer_results.append(f(xi)) # pyright: ignore[reportAny]
    concatenated = cat(embedding_layer_results, dim=1)
    return self.percentage_layer(concatenated) # pyright: ignore[reportAny]

amount = 3
data_loaders: DataLoader[tuple[Tensor, ...]] = loads(Path("data.pkl").read_bytes())

epochs = 300
model = Model(amount).to(device)
criterion = nn.HuberLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

for _ in tqdm(range(epochs)):
  _ = model.train()
  for x, y in data_loaders: # pyright: ignore[reportAny]
    x = x.to(device) # pyright: ignore[reportAny]
    y = y.to(device).unsqueeze(1) # pyright: ignore[reportAny]

    output: Tensor = model(x)
    loss = criterion(output, y) # pyright: ignore[reportAny]

    optimizer.zero_grad()
    _ = loss.backward() # pyright: ignore[reportAny]
    _ = optimizer.step() # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

save(model.state_dict(), "model.pth")