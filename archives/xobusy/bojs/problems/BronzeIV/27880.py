a = 0
for _ in range(4) :
  b, c = input().split()
  a += int(c) * [21, 17][b =='Stair']

print(a)