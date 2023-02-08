from string import ascii_lowercase

a = input()
b = [str(a.count(i)) for i in ascii_lowercase]
print(' '.join(b))
