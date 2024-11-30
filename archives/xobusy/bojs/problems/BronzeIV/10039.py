a = []

for _ in range(5):
  b = int(input())
  if b >= 40:
    a.append(b)
  else:
    a.append(40)

print(sum(a) // 5)
