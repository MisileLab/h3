_, k = map(int, input().split(" "))
lister = list(map(int, input().split(" ")))
_list = []
for i in range(1, 101):
  a = [x-i for x in lister]
  xa = sum(a)
  print(xa)
  if xa >= k:
    _list.append(xa)

print(min(_list))
