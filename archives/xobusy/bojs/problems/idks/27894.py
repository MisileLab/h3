input()
f = list(map(int, input().split(" ")))
a = list(map(int, input().split(" ")))
e = 0
_cac = len(a)-1

if a == f:
  print("POSSIBLE")
  exit()

def is_sorted(b, c, d):
  return list(sorted([b, c, d])) == [b, c, d] or list(
    sorted([b, c, d], reverse=True)
  ) == [b, c, d]

while e+2 <= _cac:
  if is_sorted(a[e], a[e+1], a[e+2]):
    print("POSSIBLE")
    break
  e += 1
else:
  print("IMPOSSIBLE")