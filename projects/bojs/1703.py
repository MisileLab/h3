a = []

def print_recursively(a: list[map]):
    for i in a:
        print(list(i))

while True:
    try:
        _list = list(map(int, input().split(" ")))
    except EOFError:
        break
    if _list == [0]:
        break
    del _list[0]
    print(_list)
    _list2 = [
        map(int, [_list[i * 2], _list[i * 2 - 1]])
        for i in range(len(_list) // 2)
    ]
    # print_recursively(_list2)
    a.append(_list2)

del a[0]

for i in a:
    leaves = 1
    for i2 in i:
        a, b = i2
        # print(a, b)
        leaves = leaves * a - b
    print(leaves)
