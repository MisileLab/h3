a = int(input())

for i in range(1, a):
  print(' ' * (a-i) + '*' * i)

for i in range(a, 0, -1):
  print(' ' * (a-i) + '*' * i)
