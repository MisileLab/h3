from math import pi
a = []

while True:
  b = list(map(float, input().split()))
  if b[1] == 0:
    break
  a.append(b)

for i2, i in enumerate(a):
  c, d, e = i
  distance = pi * c * d / 63360
  t = distance / (e / 3600)
  print(f"Trip #{i2+1}: {distance:.2f} {t:.2f}")
