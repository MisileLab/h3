a = sum(int(input()) * i for i in reversed(range(1, 4)))
b = sum(int(input()) * i for i in reversed(range(1, 4)))
if a > b:
  print("A")
elif b > a:
  print("B")
else:
  print("T")
