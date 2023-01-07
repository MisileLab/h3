from math import ceil

l = int(input())
a = []

for _ in range(4):
    a.append(int(input()))

print(f"{l-max(ceil(a[0] / a[2]), ceil(a[1] / a[3]))}")

