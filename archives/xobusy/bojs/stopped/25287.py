a = []

def get(c: list, d: int):
  if d == -1:
    return 9999
  try:
    return c[d]
  except KeyError:
    return 9999

for _ in range(int(input())):
  input()
  a.append(list(map(int, input().split(" "))))

d = False
for i in a:
  c = []
  b = i.copy()
  while d:
    for i2, i3 in enumerate(i):
      b[i2] = i3-(i2+1)+1
  
