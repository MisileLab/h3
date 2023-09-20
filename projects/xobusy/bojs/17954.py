#2n..2n-2 2n-1
#n..n-3 n-2 n-1
from copy import deepcopy
a = int(input())
b = [[2*a], [a]]

for i in range(3, a*2, 2):
    b[0].append(i)

for i in range(1, a):
    b[1].append(i)

c = 0
t = 0
f = deepcopy(b)

while len(b[1]) > 0:
    d = min(b[1])
    c += t * d
    b[1].remove(d)
    t += 1

while len(b[0]) > 0:
    d = min(b[0])
    c += t * d
    b[0].remove(d)
    t += 1

print(c)
print(*f[0])
print(*f[1])
