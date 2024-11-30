a = 0
input()

for i in list(map(int, input().split(" "))):
  a += i

if a == 0:
  print('Stay')
elif a < 0:
  print('Left')
elif a > 0:
  print('Right')

