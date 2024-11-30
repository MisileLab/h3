a, b, c = map(int, input().split(" : "))
d, e, f = map(int, input().split(" : "))
g, h = a*3600+b*60+c, d*3600+e*60+f

if g > h:
  print(24 * 3600 - g + h)
else:
  print(h - g)

