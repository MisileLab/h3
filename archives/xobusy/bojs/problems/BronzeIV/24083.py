a = int(input())
b = int(input()) % 12

if a+b > 12:
  print((a+b) % 12)
else:
  print(a+b)
