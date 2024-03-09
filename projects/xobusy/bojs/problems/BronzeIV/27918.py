a, b = 0, 0

for i in [input() for _ in range(int(input()))]:
  if abs(a-b) >= 2:
    break
  if i == "D":
    a += 1
  else:
    b += 1

print(f"{a}:{b}")
