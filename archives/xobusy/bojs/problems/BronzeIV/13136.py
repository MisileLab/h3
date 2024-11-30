from math import ceil

a, b, c = map(int, input().split(" "))
a = ceil(a/c)
b = ceil(b/c)
print(a*b)
