a, b = map(int, input().split(" "))
a = a * 1000
if b * 7000 <= a:
  c = 7000
elif b * 3500 <= a:
  c = 3500
elif b * 1750 <= a:
  c = 1750
else:
  c = 0
  print(0)

if c != 0:
  print(c * b)
