a = list(sorted(map(int, input().split(" "))))

if a[2] - (a[0] + a[1]) == 0 or a[1] == a[2] or a[0] == a[1] or a[2] == a[0]:
  print("S")
else:
  print("N")
