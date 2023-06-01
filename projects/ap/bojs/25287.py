a = []

def get(c: list, d: int):
    if d == -1:
        return 9999
    try:
        return c[d]
    except KeyError:
        return 9999

for _ in range(int(input())):
    input()
    a.append(list(map(int, input().split(" "))))

for i in a:
    b = i.copy()
    for i2, i3 in enumerate(i):
        if get(b, i2-1) > i3 - (i2+1) + 1 and get(b, i2-1) > i3:
            b[i2] = min(i3, i3-(i2+1)+1)
        elif get(b, i2-1) > i3 - (i2+1) + 1:
            b[i2] = i3 - (i2+1) + 1
        elif get(b, i2-1) > i3:
            b[i2] = i3
        else:
            break
    else:
        print("YES")
        continue
    print("NO")
