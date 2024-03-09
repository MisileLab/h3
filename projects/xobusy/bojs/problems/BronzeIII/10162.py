a = int(input())
b, c, d = (0, 0, 0)

if a % 10 != 0:
  print(-1)
else:
  while a >= 300:
    a -= 300
    b += 1
  while a >= 60:
    a -= 60
    c += 1
  while a >= 10:
    a -= 10
    d += 1
  print(f'{b} {c} {d}')