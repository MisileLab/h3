from sys import exit

a = input()
b = list('MOBIS')

for i in b:
  if a.count(i) == 0:
    print('NO')
    exit()

print('YES')
