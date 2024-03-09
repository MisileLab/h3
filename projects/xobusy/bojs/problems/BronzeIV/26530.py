a = []

for _ in range(int(input())):
  b = []
  for _ in range(int(input())):
    c = input().split(" ")
    b.append(float(c[2]) * float(c[1]))
  a.append(b)

for i in a:
  print(f"${sum(i):.2f}")
