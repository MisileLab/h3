from string import ascii_lowercase, ascii_uppercase

a = {
  'p': 1,
  'n': 3,
  'b': 3,
  'r': 5,
  'q': 9
}
b, c = 0, 0

for i in ''.join(input() for _ in range(8)):
  if i in ascii_uppercase:
    b += a.get(i.lower(), 0)
  else:
    c += a.get(i.lower(), 0)

print(b-c)

