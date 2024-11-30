a = []

while True:
  b = input()
  if b == "0 W 0":
    break
  a.append(b.split())

for i in a:
  b, c, d = int(i[0]), i[1], int(i[2])
  _tmp = b-d if c == 'W' else b+d
  if _tmp < -200:
    print("Not allowed")
  else:
    print(_tmp)
