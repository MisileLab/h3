n2 = []

n = [int(input()) for _ in range(int(input()))]
c = {}

def fibo(d: int):
  if d == 0:
    return {"a": 1, "b": 0}
  elif d == 1:
    return {"a": 0, "b": 1}
  else:
    if c.get(d) is None:
      e = [fibo(d-1), fibo(d-2)]
      c[d] = {"a": e[0]["a"] + e[1]["a"], "b": e[0]["b"] + e[1]["b"]}
    return c[d]

for i in n:
  print(f"{fibo(i)['a']} {fibo(i)['b']}")
