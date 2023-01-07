a = []

for _ in range(6):
    a.append(input())

if a.count('W') == 6 or a.count('W') == 5:
    print(1)
elif a.count('W') == 4 or a.count('W') == 3:
    print(2)
elif a.count('W') == 2 or a.count('W') == 1:
    print(3)
else:
    print(-1)
