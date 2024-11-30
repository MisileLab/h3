a, b, c = map(int, input().split())
if a % 2 != 0 and c % 2 != 0:
  c -= 1
  a -= 1

while c > 0:
  if max(a/2, b) == b:
    b -= 1
  else:
    a -= 1
  c -= 1

print(int(min(a/2, b)))
