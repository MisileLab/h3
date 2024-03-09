a = int(input())
x = []

while True:
  b = int(input())
  if b == 0:
    break
  x.append(b)

for i in x:
  if i % a == 0:
    print(f"{i} is a multiple of {a}.")
  else:
    print(f"{i} is NOT a multiple of {a}.")
