a = []
while True:
    try:
        c = list(input())
    except EOFError:
        break
    if c == 0:
        break
    a.append(c)
b = []

for i in a:
    d = 1
    for i2 in i:
        if i == "1":
            d += 3
        elif i == "0":
            d += 5
        else:
            d += 4
    b.append(d)

for i in b:
    print(i)
