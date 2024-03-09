a = [input() for _ in range(int(input()))]
b = []

for i in a:
  c = []
  for i2 in list(i):
    if c.__len__() == 0:
      c.append(i2)
    elif c[-1] != i2:
      c.append(i2)
  b.append(''.join(c))

for i in b:
  print(i)
