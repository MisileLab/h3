a, b, c = map(int, input().split(" "))
if a * b - c <= 0:
  print(0)
else:
  print(a * b - c)
