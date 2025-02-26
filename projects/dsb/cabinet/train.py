from typing import final
from sys import version_info
from pathlib import Path
from pickle import loads

if version_info < (3, 13):
  from typing_extensions import override
else:
  from typing import override

from tqdm.auto import tqdm

from torch import tensor, Tensor, nn, cat, optim, save, abs, zeros, ones, no_grad # pyright: ignore[reportUnknownVariableType]
from torch.cuda import is_available, empty_cache
from torch.utils.data import DataLoader, TensorDataset, random_split

device = "cuda" if is_available() else "cpu"
data_device = "cpu" # dataset is too big to fit in GPU memory
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

print("loading dataset")
dataset: dict[str, list[list[list[float]]]] = loads(Path("embedding.pkl").read_bytes())
print("loaded")
x_t = Tensor().to(device)
y_t = Tensor().to(device)
for i in tqdm(dataset["suicidal"]):
  t = tensor(i, device=device)
  x_t = cat((x_t, t))
  y_t = cat((y_t, ones(t.size(), device=device)))
del dataset["suicidal"]
for i in tqdm(dataset["normal"]):
  t = tensor(i, device=device)
  x_t = cat((x_t, t))
  y_t = cat((y_t, zeros(t.size(), device=device)))
del dataset
x_t = x_t.to("cpu")
y_t = y_t.to("cpu")
t_dataset = TensorDataset(x_t, y_t)
lx_t = len(x_t)
train_size = int(0.8 * lx_t)
test_size = lx_t - train_size

train_dataset, test_dataset = random_split(t_dataset, [train_size, test_size])
del t_dataset, train_size, test_size
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
  for _x, _y in train_loader:
    _x: Tensor
    _y: Tensor
    x = _x.to(device)
    y = _y.to(device)
    if x.size(0) < amount:
      continue
    optimizer.zero_grad()

    windows_x = x.unfold(0, amount, 1).transpose(1, 2)
    windows_y = y[amount - 1:]

    del x, y, _x, _y

    output: Tensor = model(windows_x)
    target = windows_y.mean(dim=1, keepdim=True)

    loss: Tensor = criterion(output, target)
    _ = loss.backward() # pyright: ignore[reportUnknownMemberType]
    _ = optimizer.step() # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    running_loss += loss.item() * windows_x.size(0)
    count += windows_x.size(0)
    del windows_x, windows_y, output, target, loss
  
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

empty_cache()

_ = model.eval()
with no_grad():
  test_x_list: list[Tensor] = []
  test_y_list: list[Tensor] = []
  for x_batch, y_batch in tqdm(test_loader, desc="Collecting test data"):
    x_batch: Tensor
    y_batch: Tensor
    _ = test_x_list.append(x_batch)
    _ = test_y_list.append(y_batch)
  test_x_all = cat(test_x_list, dim=0).to(device)
  test_y_all = cat(test_y_list, dim=0).to(device)
  
  if test_x_all.size(0) < amount:
    print("Not enough test samples to form a window.")
  else:
    windows_x = test_x_all.unfold(0, amount, 1).transpose(1, 2)
    windows_y = test_y_all[amount - 1:]
    
    predictions: Tensor = model(windows_x)
    targets = windows_y.mean(dim=1, keepdim=True)
    error = abs(predictions - targets)
    avg_error = error.mean().item()
    print("Average test error:", avg_error)
