input()
streak = 0
a = 0

for i in list(map(int, input().split(" "))):
  if i == 1:
    streak += 1
  else:
    streak = 0
  a += streak * 1

print(a)
