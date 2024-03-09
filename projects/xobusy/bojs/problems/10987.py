a = ['a','e','i','o','u']

def b(c: str):
  d = 0
  for i in c:
    if i in a:
      d += 1
  return d

print(b(input()))

