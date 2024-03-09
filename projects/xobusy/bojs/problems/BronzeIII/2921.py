a = int(input())

def b(c: int):
  e = -c
  for i in range(1, c+1): # 2
    e += i
    for i2 in range(i, c+1):
      e += i+i2
  return e

print(b(a) + a)
