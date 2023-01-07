a = int(input().split(" ")[0])
b = []

for _ in range(int(a)):
    b.append(input())

for i in b:
    print(i[::-1])

