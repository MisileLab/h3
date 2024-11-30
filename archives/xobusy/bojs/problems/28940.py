from math import ceil, floor
w, h = map(int, input().split(" "))
n, a, b = map(int, input().split(" "))
if a > w or b > h:
  print(-1)
  exit()
print(ceil(n/(floor(w/a)*floor(h/b))))
