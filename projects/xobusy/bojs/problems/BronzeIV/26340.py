from math import floor
for i in [map(int, input().split(" ")) for _ in range(int(input()))]:
  a, b, c = i
  _a, _b = a, b
  for _ in range(c):
    if max(_a, _b) == _a:
      _a = floor(_a / 2)
    else:
      _b = floor(_b / 2)
  print(f"Data set: {a} {b} {c}")
  print(max(_a, _b), min(_a, _b), end="\n\n")
