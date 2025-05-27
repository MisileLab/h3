from collections import defaultdict, deque


_ = input()
_ = input()

caches: dict[int, set[int]] = defaultdict(set)

while True:
  try:
    v, v2 = map(int, input().split(" "))
  except EOFError:
    break
  caches[v].add(v2)
  caches[v2].add(v)

current: deque[int] = deque([1])
visited: set[int] = set([1])

while current:
  v = current.popleft()
  for neighbor in caches[v]:
    if neighbor not in visited:
      visited.add(neighbor)
      current.append(neighbor)

print(len(visited) - 1)

