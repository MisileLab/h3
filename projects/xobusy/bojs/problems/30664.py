a = []

while True:
  b = int(input())
  if b == 0:
    break
  a.append(b)

for i in a:
  if i % 42 == 0:
    print("PREMIADO")
  else:
    print("TENTE NOVAMENTE")

