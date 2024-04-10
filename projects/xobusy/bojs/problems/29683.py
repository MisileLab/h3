from math import floor
_,a=map(int,input().split(" "))
print(sum(floor(x/a) for x in list(map(int, input().split(" ")))))
