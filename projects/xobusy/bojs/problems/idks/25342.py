from math import lcm

num_cases = int(input(""))
for _ in range(num_cases):
  n = int(input(""))
  tmp1 = lcm(lcm(n - 3, n - 2), n - 1)
  tmp2 = lcm(lcm(n - 2, n - 1), n)
  if n % 2 == 0:
    print(max(max(tmp1, tmp2), lcm(lcm(n - 3, n - 1), n)))
  else:
    print(max(max(tmp1, tmp2), lcm(lcm(n - 3, n - 2), n)))
