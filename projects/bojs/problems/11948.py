a = [int(input()) for _ in range(6)]
print(((a[0] + a[1] + a[2] + a[3]) - min(min(a[0], a[1]), min(a[2], a[3]))) + max(a[4], a[5]))

