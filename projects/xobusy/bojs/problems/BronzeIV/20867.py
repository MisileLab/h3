m, s, g = map(int, input().split(" "))
a, b = map(float, input().split(" "))
l, r = map(int, input().split(" "))
lwait = l / a
rwait = r / b
ls = m / g + 1 if m % g else m / g
rs = m / s + 1 if m % s else m / s
if ls + lwait < rs + rwait:
  print("friskus")
else:
  print("latmask")
