a = list(sorted(map(int, input().split(" "))))
b, c = a[0]+a[3], a[1]+a[2]
d = abs(max(b, c)-min(b, c))
print(d)
