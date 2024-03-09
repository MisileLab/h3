a = 0

for i, i2, i3 in [map(int, input().split(" ")) for _ in range(int(input()))]:
  b = i == i2
  c = i2 == i3
  d = i3 == i
  _temp = 0
  if b and c and d:
    _temp = 10000 + i * 1000
  elif b or c:
    _temp = 1000 + i2 * 100
  elif d:
    _temp = 1000 + i3 * 100
  else:
    _temp = max(i, i2 ,i3) * 100
  if max(a, _temp) == _temp:
    a = _temp

print(a)
