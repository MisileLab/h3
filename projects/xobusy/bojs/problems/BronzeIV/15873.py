a, b = 0, 0
c = False
d = input()

for i2, i in enumerate(d):
  if i != "0" and c is False:
    a += int(i)
    if d[i2 + 1] != "0":
      c = True
  elif i == "0" and c is False:
    a = 10
    c = True
  elif i != "0" and c is True:
    b += int(i)
  else:
    b = 10

print(a+b)
