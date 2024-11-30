a = []

while True:
  b = int(input())
  if b == 0:
    break
  a.append(b)

b = []

for i in a:
  b.extend("*" * i2 for i2 in range(1, i+1))
print('\n'.join(b))

