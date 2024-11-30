a, b, c = -1, -1, -1
d = [list(map(int, input().split(" "))) for _ in range(9)]

for i, i2 in enumerate(d):
  for i3, i4 in enumerate(i2):
    if i4 > a:
      a = i4
      b = i + 1
      c = i3 + 1

print(a)
print(b, c)
