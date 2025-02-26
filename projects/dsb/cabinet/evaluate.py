from typing import final, override
from os import getenv

from torch import nn, tensor, Tensor, cat, load # pyright: ignore[reportUnknownVariableType]
from torch.cuda import is_available
from openai import OpenAI

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

model = Model(amount).to(device)
_ = model.load_state_dict(load("model.pth", weights_only=True, map_location=device)) # pyright: ignore[reportAny]
_ = model.eval()
o = OpenAI(api_key=getenv("OPENAI_KEY"))
chats: list[str] = []
for i in range(3):
  _ = chats.append(input(f"chat {i+1}: "))

caches: dict[str, list[float]] = {}
embeddings: list[list[float]] = []
for i in chats:
  if caches.get(i) is None:
    resp = o.embeddings.create(
      input=i,
      model='text-embedding-3-large'
    )
    embeddings.append(resp.data[0].embedding)
    caches[i] = resp.data[0].embedding
  else:
    embeddings.append(caches[i])

print(model(tensor(embeddings).unsqueeze(0))[0][0] * 100) # pyright: ignore[reportAny]

