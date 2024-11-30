while True:
  a, b, c, d = map(int, input().split())
  if a == 0 and b == 0 and c == 0 and d == 0:
    break
  e = 0

  while not a == b == c == d:
    a1, b1, c1, d1 = a, b, c, d
    a, b, c, d = abs(a1 - b1), abs(b1 - c1), abs(c1 - d1), abs(d1 - a1)
    e += 1

  print(e)
