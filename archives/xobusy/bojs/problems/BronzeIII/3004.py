a = 2
b = 0
c = True
_prev = 0
for _ in range(int(input())):
  if b == 3:
    a += 1 if c else 2
    c = not c
    b = 0
  _prev += a
  b += 1

print(_prev)
