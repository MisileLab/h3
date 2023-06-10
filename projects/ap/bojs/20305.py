a = [0] * int(input())
b = [list(map(int, input().split())) for _ in range(int(input()))]
e = {}

FLOATING_LAND = 10 ** 9 + 7

def fibonacci(n):
    fib = [0, 1]
    fib.extend((fib[i - 1] + fib[i - 2]) % FLOATING_LAND for i in range(2, n + 1))
    return fib

fib_values = fibonacci(max(d for _, d in b))

for c, d in b:
    for i2, i3 in enumerate(range(c, d + 1)):
        a[i3 - 1] += fib_values[i2 + 1]

a = [x % FLOATING_LAND for x in a]
print(*a)
