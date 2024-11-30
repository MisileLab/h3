# sourcery skip: bin-op-identity
from math import ceil
_cache = ceil((int(input())-int(input())) / int(input()))
if _cache >= 0:
  print(250 + _cache * 100)
else:
  print(250)
