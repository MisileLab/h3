while True:
  a, b = map(int, input().split(" "))
  if a == 0 and b == 0:
    break
  c = 1
  while abs(c**b-a) >= abs((c+1)**b-a):
    c += 1
  print(c)
