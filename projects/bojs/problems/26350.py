a = []
for _ in range(int(input())):
    b = list(map(int, input().split(" ")))
    del b[0]
    a.append(b)

for i in a:
    _start = i[0]
    print(f"Denominations: {' '.join(list(map(str, i)))}")
    del i[0]
    if any(_start*2 > i2 for i2 in i):
        print("Good coin denominations!")
    else:
        print("Bad coin denominations!")
