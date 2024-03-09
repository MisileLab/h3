a, b, c = map(int, input().split(" "))
d = list(range(b, c+1))

if a in d:
  print(a)
else:
  _prev = d[0]
  _prevm = min(abs(a-d[0]), abs(d[0]-a))
  del d[0]
  for i in d:
    _temp = min(abs(a-i), abs(i-a))
    if _temp < _prevm:
      _prev = i
      _prevm = _temp
  print(_prev)
