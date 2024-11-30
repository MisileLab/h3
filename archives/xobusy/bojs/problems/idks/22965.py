input()
a = list(map(int, input().split(" ")))
b = 1
c = list(sorted(a))

while a != c and b != 3:
  b += 1
  d = []
  for i in range(2, len(a)):
    if a[i-1] > a[i]:
      d.extend(a[i2] for i2 in range(i, len(a)))
      d.extend(a[i2] for i2 in range(i))
      break
  a = d

print(b)
