for i in [map(int, input().split(" ")) for _ in range(int(input()))]:
  a, b = i
  if (a <= 1 and b <= 2) or (a <= 2 and b <= 1):
    print("Yes")
  else:
    print("No")
