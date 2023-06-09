from functools import lru_cache

a = list(range(int(input())))
b = [map(int, input().split(" ")) for _ in range(int(input()))]
e = {}

@lru_cache(maxsize=None)
def fib(n):
    if n == 0:
        return 0
    elif n in [1, 2]:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)

for i in b:
    c, d = i
    for i2, _ in enumerate(range(c, d)):
        if e.get(i2+1) is None:
            f = fib(i2)
            e[i2] = f
        else:
            f = e[i2+1]
        a[i2] += f

print(a)
