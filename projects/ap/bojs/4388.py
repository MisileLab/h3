a = []

while True:
    b = input().split()
    if b == ["0", "0"]:
        break
    a.append(b)

def get(g: list, h: int):
    try:
        return g[h]
    except IndexError:
        return 0

for i in a:
    b, c = i
    d = 0
    for i2 in range(max(len(b), len(c))):
        e, f = int(get(c, i2)), int(get(d, i2))
        if e + f >= 10:
            d += 1
            _tmp = list(b)
            try:
                _tmp[i2+1]=str(int(_tmp[i2+1])+1)
            except IndexError:
                _tmp.append(str(1))
            b = "".join(_tmp)
    print(d)