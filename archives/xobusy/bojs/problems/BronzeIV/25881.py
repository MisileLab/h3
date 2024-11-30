a, b = map(int, input().split(" "))
for i in [int(input()) for _ in range(int(input()))]:
  _base = i
  _add = 0
  if _base > 1000:
    _add = _base - 1000
    _base = 1000
  print(f"{i} {_base*a+_add*b}")
