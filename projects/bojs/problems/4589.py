a = []

for _ in range(int(input())):
  a.append(list(map(int, input().split(" "))))

print("Gnomes:")
for i in a:
  i2 = list(sorted(i))
  i3 = list(reversed(i2))
  if i == i2 or i == i3:
    print("Ordered")
  else:
    print("Unordered")
