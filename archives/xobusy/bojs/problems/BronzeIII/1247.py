for i in [[int(input()) for _ in range(int(input()))] for _ in range(3)]:
  _cache = sum(i)
  if _cache > 0:
    print("+")
  elif _cache == 0:
    print("0")
  else:
    print("-")
