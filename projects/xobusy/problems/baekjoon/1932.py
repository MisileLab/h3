n = int(input())
a: list[list[int]] = []

for _ in range(n):
  a.append(list(map(int, input().split(" "))))

for i in range(1,n):
  for j in range(i+1):
    if j==0:
      a[i][j] += a[i-1][j]
    elif j==i:
      a[i][j] += a[i-1][j-1]
    else:
      a[i][j] += max(a[i - 1][j - 1], a[i - 1][j])

print(max(a[-1]))

