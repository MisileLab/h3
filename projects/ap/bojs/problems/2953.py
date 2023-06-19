a = [sum(map(int, input().split())) for _ in range(5)]
print(a.index(max(a)) + 1, max(a))
