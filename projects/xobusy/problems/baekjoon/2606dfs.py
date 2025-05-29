from collections import defaultdict


n = int(input())
_ = input()

caches: dict[int, set[int]] = defaultdict(set)

while True:
  try:
    v, v2 = map(int, input().split(" "))
  except EOFError:
    break
  caches[v].add(v2)
  caches[v2].add(v)

visited: set[int] = set()
answer = -1

def dfs(idx: int):
  visited.add(idx)
  for i in range(1, n+1):
    if i not in visited and i in caches[idx]:
      dfs(i)

dfs(1)

print(len(visited) - 1)

