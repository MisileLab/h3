from functools import lru_cache

a = [0 for _ in range(int(input()))]
b = [map(int, input().split(" ")) for _ in range(int(input()))]
e = {}
FLOATING_LAND = 10 ** 9 + 7

@lru_cache(maxsize=None)
def fib(n):
  if n == 0:
    return 0
  elif n in [1, 2]:
    return 1
  else:
    return fib(n - 1) + fib(n - 2)

for i in b:
  c, d = i
  for i2, i3 in enumerate(range(c, d+1)):
    if e.get(i2+1) is None:
      f = fib(i2+1)
      e[i2+1] = f
    else:
      f = e[i2+1]
    a[i3-1] += f

a = [a % FLOATING_LAND for a in a]
print(*a)
