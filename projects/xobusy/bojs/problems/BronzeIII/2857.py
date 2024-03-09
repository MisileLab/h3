if a := [i+1 for i in range(5) if 'FBI' in input()]:
  print(" ".join(map(str, a)))
else:
  print("HE GOT AWAY!")
