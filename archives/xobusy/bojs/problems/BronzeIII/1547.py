ball = 1
for i in [map(int, input().split(" ")) for _ in range(int(input()))]:
  a, b = i
  if ball == a:
    ball = b
  elif ball == b:
    ball = a

print(ball)
