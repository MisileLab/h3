b = -1

for i in [map(int, input().split(" ")) for _ in range(int(input()))]:
  c, d = i
  _temp = 0
  if c <= d:
    _temp = d
  else:
    continue
  if _temp < b or b == -1:
    b = _temp

print(b)
