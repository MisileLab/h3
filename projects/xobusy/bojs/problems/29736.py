a, b = map(int, input().split(" "))
k, x = map(int, input().split(" "))

def cmp(a: int):
  return abs(a-k) <= x

c = 0
for i in range(a, b+1):
  if cmp(i):
    c += 1
print(c if c != 0 else 'IMPOSSIBLE')
