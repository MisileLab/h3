a, b = map(int, input().split(" "))
c, d = map(int, input().split(" "))
e, f = 3*a+b, 3*c+d

if e > f:
  print(f"1 {e - f}")
elif e < f:
  print(f"2 {f - e}")
else:
  print("NO SCORE")
