from math import ceil
a, b = (False, False)
c = [[],[]]
d = int(input())
e = 0
_cac = ceil((d*2) / 4)

for i in range(1, d*2+1):
    if a:
        if not b:
            c[0].insert(0, i)
        else:
            c[0].insert(-1, i)
            a = not a
    elif not b:
        c[1].insert(0, i)
    else:
        c[1].insert(-1, i)
        a = not a
    b = not b
    t = ceil(i / 4)
    e += (_cac-t) * i

if d % 2 == 1:
    c[0].insert(-1, max(c[1][0], c[1][-1]))
    c[1].remove(max(c[1][0], c[1][-1]))

print(e)
print(*c[0])
print(*c[1])
