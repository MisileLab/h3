a, b = map(int, input().split(" "))
c, d = map(int, input().split(" "))

if a+d == b+c:
  if a > c:
    print("Esteghlal")
  elif c > a:
    print("Persepolis")
  else:
    print("Penalty")
elif b+c > a+d:
  print("Esteghlal")
else:
  print("Persepolis")
