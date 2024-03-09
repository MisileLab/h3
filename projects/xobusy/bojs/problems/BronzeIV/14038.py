a = [input() for _ in range(6)]
if a.count('W') in {6, 5}:
  print(1)
elif a.count('W') in {4, 3}:
  print(2)
elif a.count('W') in {2, 1}:
  print(3)
else:
  print(-1)
