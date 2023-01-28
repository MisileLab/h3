def read_once(path: str):
    with open(path, 'r') as a:
        b = a.readlines()
    return b

a = {}

for i in read_once(".gitignore"):
    i = i.strip('\n')
    if a.get(i, None) is None:
        a[i] = 0
    else:
        a[i] += 1

for i, i2 in a.items():
    if i2 >= 1:
        print(i)
