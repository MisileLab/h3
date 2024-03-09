a = set(range(int(input())))

def getMyDivisor(n):
  divisorsList = [i for i in range(1, int(n**(1/2)) + 1) if n % i == 0]
  divisorsList += [n // i for i in divisorsList if (i**2) != n]
  divisorsList.sort()
  return divisorsList

b = True
while len(a) != 0:
  c = 0
  e = []
  for i in reversed(range(len(a))):
    if c > i:
      break
    d = [i2 for i2 in getMyDivisor(i) if i2 in a]
    c = max(c, len(a) - len(d))
    a = set(d)
  b = not b

print("A" if b else "B")
