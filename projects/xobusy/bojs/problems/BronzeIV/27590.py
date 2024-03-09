s1, s2 = map(int, input().split(" "))
m1, m2 = map(int, input().split(" "))

class a:
  def __init__(self, b: int, c: int):
    self.b = b
    self.c = c

def k():
  d = [a(s1, s2), a(m1, m2)]
  g, h = -d[0].b + d[0].c, -d[1].b + d[1].c
  e, f = [g], [h]

  while g <= 5000 or h <= 5000:
    if g <= 5000:
      g += d[0].c
      e.append(g)
    if h <= 5000:
      h += d[1].c
      f.append(h)
    i = set(e)
    for j in f:
      if j in i:
        return j

print(k())
