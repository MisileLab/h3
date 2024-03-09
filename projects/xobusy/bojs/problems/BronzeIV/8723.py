a, b, c = map(int, input().split(" "))
a2, b2, c2 = a * a, b * b, c * c

if a == b == c:
  print(2)
elif a2 + b2 == c2 or b2 + c2 == a2 or c2 + a2 == b2:
  print(1)
else:
  print(0)

