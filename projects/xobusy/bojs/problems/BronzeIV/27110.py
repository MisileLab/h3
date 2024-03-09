a = int(input())
b, c, d = map(int, input().split(" "))

def chicken(e: int, f: int):
  return f if max(e, f) == e else e

print(chicken(a, b) + chicken(a, c) + chicken(a, d))
