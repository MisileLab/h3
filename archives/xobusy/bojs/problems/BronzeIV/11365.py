a = []

while True:
  b = input()
  if b == "END":
    break
  a.append(b)

for i in a:
  print(i[::-1])
