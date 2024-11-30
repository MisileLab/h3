from string import ascii_uppercase

b = []

for i in [input() for _ in range(int(input()))]:
  _temp = list(ascii_uppercase)
  for i2 in ascii_uppercase:
    if i2 in i:
      _temp.remove(i2)
  b.append(_temp)

print("\n".join(str(i) for i in [sum(map(ord, i)) for i in b]))
