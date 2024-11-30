a, b = map(int, input().split(" "))
c = [i for i in range(1, a+1) if a%i==0]

def get(d: int):
  try:
    return c[d-1]
  except IndexError:
    return 0

print(get(b))
