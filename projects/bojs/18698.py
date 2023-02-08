a = []
b = []

for _ in range(int(input())):
    a.append(list(input()))

for i in a:
    c = 0
    for i2 in i:
        if i2 == "U":
            c += 1
        elif i2 == "D":
            break
    b.append(c)

for i in b:
    print(i)
