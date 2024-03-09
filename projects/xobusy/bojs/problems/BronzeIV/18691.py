for i in [map(int, input().split(" ")) for _ in range(int(input()))]:
  g, c, e = i
  if g == 2:
    g = 3
  elif g == 3:
    g = 5
  _cache = (e-c)*g
  if _cache < 0:
    print(0)
  else:
    print(_cache)
