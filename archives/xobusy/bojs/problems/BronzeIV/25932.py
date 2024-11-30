for i in [list(map(int, input().split(" "))) for _ in range(int(input()))]:
  print(" ".join(list(map(str, i))))
  if i.__contains__(18) and i.__contains__(17):
    print("both")
  elif i.__contains__(18):
    print("mack")
  elif i.__contains__(17):
    print("zack")
  else:
    print("none")
  print()
