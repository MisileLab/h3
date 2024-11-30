a = int(input())
b = input()
c = 0

_tmp = ["B","O","J"]
_tmp2 = 0
ps = 0
h = 0

while c != a:
  p = _tmp[_tmp2 % 3]
  if b.count(p) == 0:
    print(p, b)
    print(-1)
    exit()
  pos = b.find(p)
  print(ps)
  h += (pos-ps)**(pos-ps)
  ps += pos
  b = b[pos:]
  _tmp2 += 1

print(h)
