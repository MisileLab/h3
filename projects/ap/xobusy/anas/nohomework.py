def factorial():
    a, b = 1, 1
    while True:
        yield a+b
        a, b = b, a+b

def collatz(x: int):
    while x != 1:
        yield x
        if x % 2 == 0:
            x //= 2
        else:
            x = 3*x + 1

c, d = factorial(), collatz(int(input()))

while True:
    try:
        print(next(c), next(d))
    except StopIteration:
        break
