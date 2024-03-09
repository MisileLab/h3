for i in [
  [i3 for i3, i2 in enumerate(reversed(bin(i)[2:])) if i2 == "1"]
  for i in [int(input()) for _ in range(int(input()))]
]:
  print(*i)
