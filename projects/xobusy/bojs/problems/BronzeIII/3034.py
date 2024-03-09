from math import sqrt

a, b, c = map(int, input().split())
_c = [b, c, sqrt(b**2+c**2)]

for i in [int(input()) for _ in range(a)]:
  if i <= max(_c):
    print("DA")
  else:
    print("NE")
