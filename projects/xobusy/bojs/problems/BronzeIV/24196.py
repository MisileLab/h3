from string import ascii_uppercase

a = input()
b = 0
d = ""

while True:
  c = a[b]
  d += c
  b += ascii_uppercase.index(c)+1
  if b > len(a) - 1:
    break

print(d)
