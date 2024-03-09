n = int(input().split(" ")[0])
a = [input() for _ in range(n)]
input()
b = [input() for _ in range(n)]
c = 0

for i, i2 in zip(a, b):
  for i3, i4 in zip(i, i2):
    if i3 == i4:
      c += 1

print(c)
