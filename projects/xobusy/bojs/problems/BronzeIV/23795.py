a = 0

while True:
  try:
    b = int(input())
  except EOFError:
    break
  if b == -1:
    break
  a += b

print(a)
