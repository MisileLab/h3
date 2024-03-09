a = int(input())

for i, i2 in enumerate(range(a, 0, -1)):
  i += 1
  print('*' * i + (' ' * (i2-1))*2 + '*' * i)

for i, i2 in enumerate(range(a, 0, -1)):
  if i2 == a:
    continue
  i += 1
  print('*' * i2 + (' ' * (i-1))*2 + '*' * i2)
