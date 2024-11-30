a = [list(map(int, input().split(" "))) for _ in range(int(input()))]
d = ["".join("X"*i[0] + '\n' for _ in range(i[1])) for i in a]
_cache = len(d) - 1

for i, i2 in enumerate(d):
  if i != _cache:
    print(i2)
  else:
    print(i2, end='')
