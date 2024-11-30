n, k = map(int, input().split(" "))
li = list("B" * n)
a = 0
b = n
m = n / 2

if k > m * (n - m):
  print(-1)
  exit()
elif k == 0:
  li[-1] = "A"
  print("".join(li))
  exit()

while k > a*b:
  a += 1
  b -= 1

for i in range(a-1):
  li[i] = "A"

li[(len(li))-1-(k-((a - 1)*b))] = "A"

print("".join(li))
