a = []

for _ in range(int(input())):
  a.append(list(map(int, input().split(" "))))

for i in a:
  print(i[0] + i[1])