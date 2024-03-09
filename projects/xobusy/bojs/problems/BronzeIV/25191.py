from math import floor
chicken = int(input())
_co, _mac = map(int, input().split(" "))
total = _mac + floor(_co / 2)
if total <= chicken:
  print(total)
else:
  print(chicken)
