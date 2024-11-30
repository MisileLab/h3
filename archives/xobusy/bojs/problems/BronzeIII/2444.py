a = int(input())

for i, i2 in enumerate(range(a, 0, -1)):
  print(' ' * (i2-1) + '*' * ((i+1) * 2 - 1))

for i, i2 in enumerate(range(a, 0, -1)):
  print(' ' * (i+1) + '*' * ((i2-1) * 2 - 1))
