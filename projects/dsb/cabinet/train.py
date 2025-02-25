from typing import final
from sys import version_info
from pathlib import Path
from pickle import loads

if version_info < (3, 13):
  from typing_extensions import override
else:
  from typing import override

from tqdm.auto import tqdm

from torch import tensor, Tensor, nn, cat, optim, save, abs # pyright: ignore[reportUnknownVariableType]
from torch.cuda import is_available
from torch.utils.data import DataLoader, TensorDataset, random_split

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
batch_size = 32

dataset: dict[str, list[list[list[float]]]] = loads(Path("embedding.pkl").read_bytes())
x_t = Tensor(device=device)
y_t = Tensor(device=device)
for i in dataset["suicidal"]:
  x_t = cat((x_t, tensor(i, device=device)))
  y_t = cat((y_t, tensor([1], device=device)))
for i in dataset["normal"]:
  x_t = cat((x_t, tensor(i, device=device)))
  y_t = cat((y_t, tensor([0], device=device)))
t_dataset = TensorDataset(x_t, y_t)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(t_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

epoches = 300
model = Model(amount).to(device)
criterion = nn.HuberLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

t = tqdm(range(epoches))

# Early Stopping Parameters
patience = 5
best_loss = float('inf')
patience_counter = 0
model_path = "model.pth"
best_model = model.state_dict()

for epoch in t:
  _ = model.train()
  running_loss = 0.0
  count = 0
  for x, y in train_loader:
    n = amount-1
    if len(x) < amount:
      continue
    sub_x_t = Tensor(device=device)
    sub_y_t = Tensor(device=device)
    while n < len(x):
      sub_x_t = cat((sub_x_t, x[n-amount:n]))
      sub_y_t = cat((sub_y_t, y[0]))
      n += 1
      count += 1
    x: Tensor
    y: Tensor
    x = x.to(device)
    y = y.to(device).unsqueeze(1)
    optimizer.zero_grad()

    output: Tensor = model(x)
    loss = criterion(output, y) # pyright: ignore[reportAny]

    _ = loss.backward() # pyright: ignore[reportAny]
    _ = optimizer.step() # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    running_loss += loss.item() # pyright: ignore[reportAny]
  
  avg_loss = running_loss / count
  t.set_description(str(avg_loss))
  
  if avg_loss < best_loss:
    best_loss = avg_loss
    patience_counter = 0
    best_model = model.state_dict()
  else:
    patience_counter += 1
    if patience_counter >= patience:
      print("Early stopping triggered!")
      break

print("Training completed!")

_ = save(best_model, model_path)
_ = model.load_state_dict(best_model)

_ = model.eval()
x_test: Tensor
y_test: Tensor
res: list[Tensor] = []
for x_test, y_test in test_loader:
  x_test, y_test = x_test.to(device), y_test.to(device).unsqueeze(1)
  prediction: Tensor = model(x_test)
  _ = res.append(abs(prediction - y_test))
print(sum(res) / len(res))
