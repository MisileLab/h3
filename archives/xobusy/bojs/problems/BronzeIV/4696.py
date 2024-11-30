a = []

while True:
  b = float(input())
  if b == 0:
  break
  a.append(b)

for i in a:
  print(f"{float(1 + i + i ** 2 + i ** 3 + i ** 4):.2f}")
