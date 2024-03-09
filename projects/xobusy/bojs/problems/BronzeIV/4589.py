print("Gnomes:")
for i in [list(map(int, input().split(" "))) for _ in range(int(input()))]:
  i2 = list(sorted(i))
  i3 = list(reversed(i2))
  if i in [i2, i3]:
  print("Ordered")
  else:
  print("Unordered")
