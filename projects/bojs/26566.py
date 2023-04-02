from math import pi

for i in [
    [map(int, input().split(" ")), map(int, input().split(" "))]
    for _ in range(int(input()))
]:
    w1, w2 = i[0]
    s1, s2 = i[1]
    r = ((s1 / (2 * pi)) ** 2 * pi) / s2
    r2 = w1 / w2
    print((s1 / (2 * pi)) ** 2 * pi)
    print(r2)
    print(r)
