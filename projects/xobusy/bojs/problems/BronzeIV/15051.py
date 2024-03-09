c = 0

a = [int(input()) for _ in range(3)]
for i in range(1, 4):
  b = sum(abs(i2+1 - i) * i3 * 2 for i2, i3 in enumerate(a))
  c = b if c == 0 else min(b, c)
print(c)
