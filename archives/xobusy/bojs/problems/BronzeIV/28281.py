n, x = map(int, input().split(" "))
a = list(map(int, input().split(" ")))
_list = [a[i-1] * x + a[i] * x for i in range(1, n)]
print(min(_list))
