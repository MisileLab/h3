a, b = int(input()), int(input())

c = []

for _ in range(a):
    c.append("*" * b)

print('\n'.join(c), end='')
