a, b, c = map(int, input().split(" "))

if a+b+c >= 100:
  print("OK")
else:
  minn = min(a, min(b, c))
  if minn == a:
    print("Soongsil")
  elif minn == b:
    print("Korea")
  else:
    print("Hanyang")
