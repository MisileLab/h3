input()
a = 0
b = list(map(int, input().split(" ")))

for i, i2 in enumerate(b):
  if i != len(b)-1:
    a += 8
  a += i2

print(a // 24, a % 24)
