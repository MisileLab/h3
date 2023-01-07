a = []

while True:
    b = int(input())
    if b == 0:
        break
    a.append(b)

b = []

for i in a:
    for i2 in range(1, i+1):
        b.append("*" * i2)

print('\n'.join(b))

