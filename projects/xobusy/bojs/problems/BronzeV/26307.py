a, b = map(int, input().split(" "))
b += a * 60
del a

print(-(540 - b))
