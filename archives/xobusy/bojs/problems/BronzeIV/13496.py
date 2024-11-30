a = []
e = []

for _ in range(int(input())):
  n, s, d = map(int, input().split(" "))
  b = {"s": s, "d": d, "data": []}

  for _ in range(n):
    c, d = map(int, input().split(" "))
    b['data'].append([c, d])

  a.append(b)

for i in a:
  f = i['d'] * i['s']
  m = sum(i2[1] for i2 in i['data'] if i2[0] - f <= 0)
  e.append(m)

for i, i2 in enumerate(e):
  if i+1 != len(e):
    print(f"Data Set {i+1}:\n{i2}\n")
  else:
    print(f"Data Set {i+1}:\n{i2}")
