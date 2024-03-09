a = int(input())
b = int(input()) + 60

if a <= b:
  print(a * 1500)
else:
  print(b * 1500 + (a-b) * 3000)
