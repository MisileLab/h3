from math import floor
n, n2 = map(int, input().split(" "))
a, b = 100-n, 100-n2
c, d = 100-(a+b), a*b
q, r = floor(d/100), d%100
print(a, b, c, d, q, r)
print(c+q, r)
