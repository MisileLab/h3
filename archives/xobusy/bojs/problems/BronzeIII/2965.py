a = sorted(map(int, input().split()))
print(max(a[2]-a[1]-1, a[1]-a[0]-1))
