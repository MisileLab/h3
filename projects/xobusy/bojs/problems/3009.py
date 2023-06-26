a = [tuple(map(int, input().split())) for _ in range(3)]
xlist = [x for x, _ in a]
ylist = [y for _, y in a]
x = [x for x in xlist if xlist.count(x) % 2 != 0][0]
y = [y for y in ylist if ylist.count(y) % 2 != 0][0]
print(x, y)
