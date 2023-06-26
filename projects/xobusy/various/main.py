from orjson import dumps, loads
from misilelibpy import read_once

a = loads(read_once('asd.json'))
b = list(a[0].keys())
del b[b.count('diagnosis') - 1]
c = {}

for i2 in b:
    for i in a:
        if c.get(i2[i]) is None:
            c[i2[i]] = 0
        if i['diagnosis'] == 1:
            c[i2[i]] += 1
    print(f"{i2}: {c[i2]}")
