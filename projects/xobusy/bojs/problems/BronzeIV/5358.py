from sys import stdin

a = []

for i in stdin:
  if i == "":
    break
  a.append(i.strip('\n'))

for i in a:
  print(i.replace('i', '1').replace('I', '2').replace('e', 'i').replace('E', 'I').replace('1', 'e').replace('2', 'E'))
