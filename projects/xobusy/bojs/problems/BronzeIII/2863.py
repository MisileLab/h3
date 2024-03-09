a, b = map(int, input().split())
c, d = map(int, input().split())

def rotate():
  global a, b, c, d
  _a, _b, _c, _d = a, b, c, d
  a, b, c, d = _c, _a, _d, _b

mins = [a/c+b/d]

for _ in range(3):
  rotate()
  mins.append(a/c+b/d)

print(mins.index(max(mins)))
