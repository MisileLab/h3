a=[input() for _ in range(int(input()))]
for i, j in [map(int, input().split(" ")) for _ in range(int(input()))]:
 print(*a[i-1:j], sep="\n")
