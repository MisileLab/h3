a = int(input())
b = list(map(int, input().split(" ")))

if [x for x in b if x > 0].__len__() == 0:
  print(0)
  exit()

c = b.copy()
c.extend(b)
c = [x - (i+1) for i, x in enumerate(c)]

_min = c[0]
i = 1
_list = []
while _min > c[i]:
  _list.append(_min)
  _min = c[i]
  i += 1
_list.append(c[i])
i += 1
while _min < c[i]:
  _list.append(_min)
  _min = c[i]
  i += 1
_list.append(_min)
print(sum(_list))