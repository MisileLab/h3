def t(n: int):
  return sum(range(1, n+1))

for i in [int(input()) for _ in range(int(input()))]:
  print(sum(k*t(k+1) for k in range(1, i+1)))
