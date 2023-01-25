a, b, c = map(int, input().split(" "))

print(min(b, c) + min(a-b, a-c))
