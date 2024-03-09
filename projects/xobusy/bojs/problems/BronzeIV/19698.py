from math import floor
a, b, c, d = map(int, input().split(" "))
e = floor(b / d) * floor(c / d)
if e > a:
  print(a)
else:
  print(e)
