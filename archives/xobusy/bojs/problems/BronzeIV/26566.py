from math import pi

for i in [
  [map(int, input().split(" ")), map(int, input().split(" "))]
  for _ in range(int(input()))
]:
  w1, w2 = i[0]
  s1, s2 = i[1]
  r = pi * (s1 ** 2) / s2
  r2 = w1 / w2
  if r < r2:
    print("Slice of pizza")
  else:
    print("Whole pizza")
