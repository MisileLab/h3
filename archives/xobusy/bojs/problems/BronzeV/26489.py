from sys import stdin

a = 0

for i in stdin:
  if i == "":
    break
  a += 1

print(a)