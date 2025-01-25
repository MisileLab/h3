from pathlib import Path
from collections import defaultdict
from json import loads
from typing import final, override

from torch import Tensor, nn, cat, optim

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

normals = Path("embedding_results").glob("normal*")
suicidals = Path("embedding_results").glob("suicidal*")
processed_ids: dict[int, list[int]] = defaultdict(list)

model = Model(amount)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())
_ = model.train()

# TODO: make entire data to stream of Tensor

# TODO: actual train

