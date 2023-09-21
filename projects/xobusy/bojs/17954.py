#2n-3, 2n-4, 2n-5...n-1, 2n-2
#2n-1, n-2, n-3...1, 2n
from contextlib import suppress

def suppressor(i, i2):
    with suppress(ValueError):
        i.remove(i2)

a = int(input())
c = 2*a
_tmp = [[c-3,c-4,c-5], [c-1,a-2,a-3]]
b = [[i for i in _tmp[0] if i > 0], [i for i in _tmp[1] if i > 0]]
excluded = [c-1,c-2,c-3,c-4,c-5,a-2,a-3,1,a-1]
i = 0

_tmp = c
_cac = _tmp-i
while _cac >= a:
    if _cac not in excluded:
        b[0].append(_cac)
    i += 1
    _cac = _tmp-i

i = 0
_tmp = a
_cac = _tmp-i
while _cac > 0:
    if _cac not in excluded:
        b[0].append(_cac)
    i += 1
    _cac = _tmp-i

for i in [_tmp - 1, c-2]:
    if i > 0:
        suppressor(b[0], i)
        suppressor(b[1], i)
        b[0].append(i)

for i in [1, c]:
    suppressor(b[0], i)
    suppressor(b[1], i)
    b[1].append(i)

# check if the length of b[0] and b[1] are not equal and make them equal
if len(b[0]) != len(b[1]):
    if len(b[0]) > len(b[1]):
        diff = len(b[0]) - len(b[1])
        for _ in range(diff):
            b[1].append(b[0].pop())
    else:
        diff = len(b[1]) - len(b[0])
        for _ in range(diff):
            b[0].append(b[1].pop())

print(b)
