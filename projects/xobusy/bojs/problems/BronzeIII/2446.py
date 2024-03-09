a = int(input())

for i, i2 in enumerate(range(a+1, 0, -1)):
  if i == a-1:
    break
  print(' ' * i + '*' * ((i2-1) * 2 - 1))

for i, i2 in enumerate(range(a, 0, -1)):
  print(' ' * (i2-1) + '*' * ((i+1) * 2 - 1))
