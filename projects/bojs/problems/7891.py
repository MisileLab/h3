a = []

for _ in range(int(input())):
    a.append(input().split(" "))

for i in a:
    b, c = map(int, i)
    print(b+c)
