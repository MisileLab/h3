a = input()
a = list(map(int, [a[0], a[4], a[8]]))

if a[0] + a[1] == a[2]:
  print("YES")
else:
  print("NO")
