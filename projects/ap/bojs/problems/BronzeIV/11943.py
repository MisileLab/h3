a = list(map(int, input().split(" ")))
b = list(map(int, input().split(" ")))

c = min(a[0]+b[1], a[1]+b[0])

print(c)
