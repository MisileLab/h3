from math import floor

a, b, c = map(int, input().split(" "))
print(floor(a*b/c) if a*b/c>a/b*c else floor(a/b*c))
