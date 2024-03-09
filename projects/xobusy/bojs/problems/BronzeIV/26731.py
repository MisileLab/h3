from string import ascii_uppercase

a = list(input())

for i in ascii_uppercase:
  if i not in a:
    print(i)
    break
