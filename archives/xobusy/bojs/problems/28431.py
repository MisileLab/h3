a = {}

for _ in range(5):
  b = int(input())
  if a.get(b) is None:
    a[b] = 0
  a[b] += 1

for i, i2 in a.items():
  if i2 % 2 == 1:
    print(i)
    break

