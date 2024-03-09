a = [list(map(float, input().split(" "))) for _ in range(int(input()))]

for i in a:
  print("${:.2f}".format(round(i[0] * (i[1] * i[2]), 2)))