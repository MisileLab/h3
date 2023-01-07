a = []
c = 0

for _ in range(3):
    a.append(int(input()))

for i in range(1, 4):
    b = 0
    for i2, i3 in enumerate(a):
        b += abs(i2+1 - i) * i3 * 2
    if c == 0:
        c = b
    else:
        c = min(b, c)

print(c)
