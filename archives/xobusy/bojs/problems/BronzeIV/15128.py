from fractions import Fraction

a, b, c, d = map(int, input().split(" "))
e, f = Fraction(a, b), Fraction(c, d)

if float(e * f) / 2 == int(float(e * f) / 2):
  print(1)
else:
  print(0)
