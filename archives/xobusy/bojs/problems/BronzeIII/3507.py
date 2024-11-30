from itertools import product
a = int(input())
print(sum(a - i - i2 == 0 for i, i2 in product(range(100), range(100))))
