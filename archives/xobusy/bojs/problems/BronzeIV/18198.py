a, b = (0, 0)
c = input()
_prev = ""

for i in c:
  if _prev == "A":
    a += int(i)
  elif _prev == "B":
    b += int(i)
  _prev = i

if a > b:
  print("A")
else:
  print("B")
