from math import floor

a, b, c = map(int, input().split(" "))

if a*b/c>a/b*c:
  d = floor(a*b/c)
else:
  d = floor(a/b*c)

print(d)