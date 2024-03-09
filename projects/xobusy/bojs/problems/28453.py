input()

a = []
for i in list(map(int, input().split(" "))):
  if i >= 300:
    a.append(1)
  elif i >= 275:
    a.append(2)
  elif i >= 250:
    a.append(3)
  else:
    a.append(4)

print(*a)

