#2n..2n-2 2n-1
#n..n-3 n-2 n-1
from copy import deepcopy
a, b = (False, False)
c = [[],[]]
d = int(input())
e = 0

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

if d % 2 == 1:
    c[0].insert(-1, max(c[1][0], c[1][-1]))
    c[1].remove(max(c[1][0], c[1][-1]))
t = 1
answer = deepcopy(c)

while len(c[0]) != 0 or len(c[1]) != 0:
    try:
        f = min(c[0][0], c[0][-1])
    except IndexError:
        g = min(c[1][0], c[1][-1])
        c[1].remove(g)
        for i in c:
            for i2 in i:
                e += t * i2
        continue
    try:
        g = min(c[1][0], c[1][-1])
    except IndexError:
        c[0].remove(f)
        for i in c:
            for i2 in i:
                e += t * i2
        continue
    if f <= g:
        c[0].remove(f)
    else:
        c[1].remove(g)
    for i in c:
        for i2 in i:
            e += t * i2
    t += 1

print(e)
print(*answer[0])
print(*answer[1])
