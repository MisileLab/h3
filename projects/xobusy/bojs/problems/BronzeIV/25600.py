_point = 0

for i in [map(int, input().split(" ")) for _ in range(int(input()))]:
  a, d, g = i
  _temp = a * (d+g)
  if a == d+g:
    _temp *= 2
  if _temp > _point:
    _point = _temp

print(_point)
