from math import floor
a, b = map(int, input().split(" "))
c1, c2 = floor((a - 1) / 4) + 1, floor((a-1) % 4)
d1, d2 = floor((b - 1) / 4) + 1, floor((b-1) % 4)
print(abs(c1-d1) + abs(c2-d2))
