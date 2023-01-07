a = int(input())
b = 0
c = list(map(int, input().split(" ")))

for i in c:
    if i == a:
        b += 1

print(b)
