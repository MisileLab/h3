for i in [int(input()) for _ in range(int(input()))]:
  a = i // 25
  i %=25
  b = i // 10
  i %=10
  c = i // 5
  i %= 5
  print(a, b, c, i)
