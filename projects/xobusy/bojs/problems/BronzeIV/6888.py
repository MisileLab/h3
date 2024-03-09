a = int(input())
b = int(input())
c = []

while a <= b:
  c.append(a)
  if a + 60 <= b:
    a += 60
  else:
    break

for i in c:
  print(f"All positions change in year {i}")
