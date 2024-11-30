a = input()
b, c = 0, 0

if min(a.count("0"), a.count("1")) == 0:
  print(0)
  exit()

_tmp = ""

for i in a:
  if _tmp != i:
    _tmp = i
    if _tmp == '0':
      b += 1
    else:
      c += 1

print(min(b, c))
