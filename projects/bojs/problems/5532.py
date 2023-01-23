from math import ceil

l = int(input())
a = [int(input()) for _ in range(4)]
print(f"{l-max(ceil(a[0] / a[2]), ceil(a[1] / a[3]))}")
