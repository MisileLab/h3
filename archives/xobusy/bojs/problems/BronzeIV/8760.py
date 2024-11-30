from math import floor

for i, i2 in [map(int, input().split(" ")) for _ in range(int(input()))]:
  print(floor((i * i2) / 2))
