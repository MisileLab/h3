from math import floor

a, b = map(int, input().split(" "))

while b > 0:
  a = floor(a / 2) ^ 6 if a % 2 == 0 else 2*a^6
  b -= 1

print(a)
