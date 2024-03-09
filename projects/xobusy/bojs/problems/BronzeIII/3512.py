bedroom = 0
bal = 0
common = 0

a, b = map(int, input().split(" "))

for _ in range(a):
  c, d = map(str, input().split(" "))
  c = int(c)
  if d == "bedroom":
    bedroom += c
  elif d == "balcony":
    bal += c
  else:
    common += c

print(common + bedroom + bal)
print(bedroom)
print(b * (common + bedroom + bal / 2))
