from math import gcd

def lcm(c: list):
  lcm = 1
  for i in c:
    lcm = lcm*i//gcd(lcm, i)
  return lcm

a = int(input())
b = list(map(int, input().split(" ")))
print(lcm(b))
