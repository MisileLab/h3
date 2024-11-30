
for i in [map(str, input().split(" ")) for _ in range(int(input()))]:
  a, b, c = i
  b, c = (int(b), int(c))
  a = list(a)
  del a[b:c]
  print("".join(a))
