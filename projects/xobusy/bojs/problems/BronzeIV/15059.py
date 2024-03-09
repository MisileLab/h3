a, b, c = map(int, input().split(" "))
d, e, f = map(int, input().split(" "))

def minus(n: int, n2: int):
  return max(n - n2, 0)

print(minus(d, a)+minus(e, b)+minus(f, c))

