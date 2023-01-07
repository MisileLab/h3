from string import ascii_lowercase

a = input()
b = []

for i in ascii_lowercase:
    b.append(str(a.count(i)))

print(' '.join(b))
