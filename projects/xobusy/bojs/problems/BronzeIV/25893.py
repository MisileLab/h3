_list = [list(map(int, input().split(" "))) for _ in range(int(input()))]
for i2, i in enumerate(_list):
  _min = sum(min(i3, 10) == 10 for i3 in i)
  print(' '.join(list(map(str, i))))
  if _min == 0:
    print("zilch")
  elif _min == 1:
    print("double")
  elif _min == 2:
    print("double-double")
  else:
    print("triple-double")
  if i2 != len(_list)-1:
    print()
