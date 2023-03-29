
for i in [map(str, input().split(" ")) for _ in range(int(input()))]:
    a, b, c = i
    b, c = (int(b), int(c))
    a = list(a)
    print(len(a))
    for i in range(b, c):
        del a[i]
    print("".join(a))
