from math import ceil

a, b = map(int, input().split(" "))
c = min(max(ceil(a / 2), a-ceil(a / 2)) * b, max(ceil(b / 2), b-ceil(b / 2)) * a)

print((a*b - (a*b - c)) - (a*b - c))

