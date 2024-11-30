a = [int(input()) for _ in range(5)]
c = 0

if a[0] < 0:
  c = (abs(a[0]) * a[2])
  a[0] = 0

if a[0] == 0:
  c += a[3]

c += (a[1] - a[0]) * a[4]
print(c)
