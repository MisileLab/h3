from math import pow, cbrt
print(cbrt(sum(pow(i, 3) for i in list(map(float, input().split(" "))))))
