y, m = 0, 0
input()

for i in list(map(int, input().split(" "))):
  y += ((i//30)+(i%29!=0))*10
  m += ((i//60)+(i%59!=0))*15

if y > m:
  print("M", m)
elif y == m:
  print("Y M", m)
else:
  print("Y", y)
