from math import floor

a = []

for _ in range(int(input())):
    a.append(map(int, input().split(" ")))

for i, i2 in a:
    print(floor((i * i2) / 2))
