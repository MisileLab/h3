a = []

def try_get(b: int):
  b = max(b, 0)
  try:
    return a[b]
  except IndexError:
    return 0

for i3, (i, i2) in enumerate(map(int, input().split()) for _ in range(4)):
  c = try_get(i3-1)
  c -= i
  c += i2
  a.append(c)

print(max(a))
