from math import pi

a, b = map(int, input().split(" "))
c, d = map(int, input().split(" "))

if a / b > c ** 2 * pi / d:
  print("Slice of pizza")
else:
  print("Whole pizza")
