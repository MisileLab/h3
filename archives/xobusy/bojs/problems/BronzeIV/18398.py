a = []
b = []

for _ in range(int(input())):
  for _ in range(int(input())):
    c, d = map(int, input().split(" "))
    a.append(c)
    b.append(d)

for i, i2 in zip(a, b):
  print(f"{i+i2} {i*i2}")
