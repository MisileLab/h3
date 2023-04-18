from sys import argv

def read_once(path: str):
    # file deepcode ignore PT: <please specify a reason of ignoring this>
    with open(path, 'r') as a:
        b = a.readlines()
    return b

a = {}

for i in read_once(argv[1]):
    i = i.strip('\n')
    if a.get(i) is None:
        a[i] = 0
    else:
        a[i] += 1

for i, i2 in a.items():
    if i2 >= 1:
        print(i)
