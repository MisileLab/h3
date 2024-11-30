a = []

while True:
  b = input().split()
  if b == ["0", "0", "0"]:
    break
  a.append(sorted(list(map(int, b))))

for i in a:
  if i[0]**2 + i[1]**2 == i[2]**2:
    print("right")
  else:
    print("wrong")