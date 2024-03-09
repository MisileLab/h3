a, b = map(int, input().split(" "))
c = (a-b) // 2
if c+b < 0 or c < 0:
  print(-1)
else:
  print(c + b, c)
