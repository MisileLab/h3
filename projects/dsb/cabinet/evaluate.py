from typing import final, override
from os import getenv

from torch import nn, Tensor, cat, load # pyright: ignore[reportUnknownVariableType]
from openai import OpenAI

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

model = Model(amount)
_ = model.load_state_dict(load('model.pth', weights_only=True)) # pyright: ignore[reportAny]
_ = model.eval()
o = OpenAI(api_key=getenv("OPENAI_KEY"))
chats: list[str] = []
for i in range(3):
  _ = chats.append(input(f"chat {i+1}: "))

embeddings: list[Tensor] = []
for i in chats:
  resp = o.embeddings.create(
    input=i,
    model='text-embedding-3-large'
  )
  embeddings.append(Tensor(*resp.data[0].embedding))

print(model(cat(embeddings, dim=0))) # pyright: ignore[reportAny]

