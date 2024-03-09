input()
a = 0
b = 0
for i in list(map(int, input().split(" "))):
  if i % 2 == 0:
    a += 1
  else:
    b += 1
if a > b:
  print("Happy")
else:
  print("Sad")
