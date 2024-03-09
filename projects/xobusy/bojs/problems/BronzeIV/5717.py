a = []

while True:
  b = list(map(int, input().split(" ")))
  if b == [0, 0]:
    break
  a.append(b)

for i in a:
  print(sum(i))
