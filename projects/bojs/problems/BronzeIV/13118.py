b = list(map(int, input().split(" ")))
x, _, _ = map(int, input().split(" "))

print(b.index(x)+1 if x in b else 0)
