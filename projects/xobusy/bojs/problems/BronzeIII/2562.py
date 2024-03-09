a, b = -1, -1

for i, i2 in enumerate(int(input()) for _ in range(9)):
  if i2 > b:
    a, b = i, i2

print(b)
print(a+1)
