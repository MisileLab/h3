a, b = map(int, input().split(" "))
_cache = a*7
_cache2 = b*13
if _cache > _cache2:
  print("Axel")
elif _cache == _cache2:
  print("lika")
else:
  print("Petra")
