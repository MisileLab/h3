a, b, c = map(int, input().split(" "))
_avg = sorted([a, b, c])[2]
_min = min(a, b, c)
_max = max(a, b, c)
print(abs(_avg-_min)+abs(_avg-_max))
