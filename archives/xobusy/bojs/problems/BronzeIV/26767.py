for i in range(1, int(input())+1):
  a, b = i % 7 == 0, i % 11 == 0
  if a and b:
    print("Wiwat!")
  elif b:
    print("Super!")
  elif a:
    print("Hurra!")
  else:
    print(i)
