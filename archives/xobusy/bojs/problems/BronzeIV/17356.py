a, b = map(int, input().split(" "))

m = (b - a)/400
print(1/(1+(10 ** ((b - a) / 400))))
