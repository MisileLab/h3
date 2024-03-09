a = [int(input()) for _ in range(7)]
if a := [i for i in a if i % 2 == 1]:
  print(sum(a))
  print(min(a))
else:
  print(-1)
