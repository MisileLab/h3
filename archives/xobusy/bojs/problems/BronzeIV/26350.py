a = []
for _ in range(int(input())):
  b = list(map(int, input().split(" ")))
  del b[0]
  a.append(b)
_cache = len(a)-1

for i3, i in enumerate(a):
  _start = i[0]
  print(f"Denominations: {' '.join(list(map(str, i)))}")
  del i[0]
  _end = "\n\n"
  if i3 == _cache:
    _end = "\n"
  for i2 in i:
    if _start*2 > i2:
      print("Bad coin denominations!", end=_end)
      break
    _start = i2
  else:
    print("Good coin denominations!", end=_end)
