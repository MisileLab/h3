a, b, c = map(int, input().split(" "))

if a == b and b == c and c == a:
  print("*")
elif a != b and a != c and b == c:
  print("A")
elif b != a and b != c and a == c:
  print("B")
else:
  print("C")
