a=0
n,_=map(int, input().split(" "))

for i in [input() for _ in range(n)]:
 if '+' in i:
  a += 1

print(a)

