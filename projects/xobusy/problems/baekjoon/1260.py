from contextlib import suppress
from collections import defaultdict, deque


n, _, start = map(int, input().split(" "))

caches: dict[int, set[int]] = defaultdict(set)

def bfs(idx: int):
  current: deque[int] = deque([idx])
  visited: list[int] = [idx]

  while current:
    v = current.popleft()
    for neighbor in sorted(caches[v]):
      if neighbor not in visited:
          visited.append(neighbor)
          current.append(neighbor)

  return visited


def dfs(idx: int):
  visited: list[int] = []

  def _dfs(v: int):
    visited.append(v)
    for neighbor in sorted(caches[v]):
      if neighbor not in visited:
        _dfs(neighbor)

  _dfs(idx)
  return visited

with suppress(EOFError):
  while True:
    v, v2 = map(int, input().split(" "))
    caches[v].add(v2)
    caches[v2].add(v)

print(" ".join(map(str, dfs(start))))
print(" ".join(map(str, bfs(start))))

