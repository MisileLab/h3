a, b, c = map(int, input().split(" "))
d, e, f = map(int, input().split(" "))

def minus(n: int, n2: int):
    n3 = n - n2
    if n3 > 0:
        return n3
    else:
        return 0

print(minus(d, a)+minus(e, b)+minus(f, c))

