from math import floor

for i in [map(int, input().split(" ")) for _ in range(int(input()))]:
  a, b = i
  _live, _dead = floor(b / 4), floor(b / 7)
  print(a+(_live-_dead))
