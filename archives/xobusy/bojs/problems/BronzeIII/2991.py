a, b, c, d = map(int, input().split())
e, f, g = map(int, input().split())

def normal_sum(h: int):
  u = 0
  if h % (a+b) <= a and h % (a+b) != 0:
    u += 1
  if h % (c+d) <= c and h % (c+d) != 0:
    u += 1
  return u

print(normal_sum(e), normal_sum(f), normal_sum(g), sep='\n')
