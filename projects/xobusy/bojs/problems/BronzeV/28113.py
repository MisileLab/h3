from math import inf

a, b, c = map(int, input().split(" "))
d = a-(a-c)

if b < d:
  print("Bus")
elif b == d:
  print("Anything")
else:
  print("Subway")
