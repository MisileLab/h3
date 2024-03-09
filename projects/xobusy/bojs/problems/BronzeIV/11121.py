a = []

for _ in range(int(input())):
  b, c = map(str, input().split(" "))
  if b == c:
    a.append("OK")
  else:
    a.append("ERROR")

for i in a:
  print(i)
