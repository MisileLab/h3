a = []
n = int(input())
for _ in range(n):
  x, y = map(int, input().split())
  a.append((x, y))

candidate1, candidate2 = 0, 0
maxval, minval = float('-inf'), float('inf')

for i in range(n):
  maxval = max(maxval, a[i][0] + a[i][1])
  minval = min(minval, a[i][0] + a[i][1])

candidate1 = maxval - minval

maxval, minval = float('-inf'), float('inf')

for i in range(n):
  maxval = max(maxval, a[i][0] - a[i][1])
  minval = min(minval, a[i][0] - a[i][1])

candidate2 = maxval - minval

print(max(candidate1, candidate2))
