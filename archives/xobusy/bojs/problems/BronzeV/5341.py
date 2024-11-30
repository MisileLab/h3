a = []

while True:
  b = int(input())
  if b == 0:
    break
  a.append(b)

for i in a:
  print(sum(range(1, i+1)))
