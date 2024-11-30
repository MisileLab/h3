a, b, c = map(int, input().split(" "))
if c*b+a == a*b:
  print(0)
elif a / 2 < c:
  print(1)
else:
  print(2)
