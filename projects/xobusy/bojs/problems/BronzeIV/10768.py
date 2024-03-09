a, b = map(int, [input(), input()])

if a == 2 and b == 18:
  print("Special")
elif a < 2 or (a == 2 and b < 18):
  print("Before")
elif a > 2 or (a == 2 and b > 18):
  print("After")
