b = []

for i in [list(input()) for _ in range(int(input()))]:
  c = 0
  for i2 in i:
    if i2 == "D":
      break
    elif i2 == "U":
      c += 1
  b.append(c)

for i in b:
  print(i)
