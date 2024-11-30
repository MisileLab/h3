a = []

for _ in range(int(input())):
  input()
  n = int(input())
  b = [n]
  b.extend(int(input()) for _ in range(n))
  a.append(b)

for i in a:
  n = i[0]
  del i[0]
  if sum(i) % n == 0:
    print("YES")
  else:
    print("NO")
