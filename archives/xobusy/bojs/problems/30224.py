a = int(input())

if '7' not in str(a):
  if a % 7 != 0:
    print(0)
  else:
    print(1)
elif a % 7 != 0:
  print(2)
else:
  print(3)

