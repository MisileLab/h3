a = []
while True:
  try:
    c = int(input())
  except EOFError:
    break
  if c == 0:
    break
  a.append(c)
b = []

for i in a:
  d = 1
  while i != 0:
    _cache = i%10
    if _cache == 0:
      d += 5
    elif _cache == 1:
      d += 3
    else:
      d += 4
    i = int(i / 10)
  b.append(d)

for i in b:
  print(i)
