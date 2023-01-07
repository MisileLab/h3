a = []

for _ in range(4):
    a.append(int(input()))

if a[0] in [8, 9] and a[1] == a[2] and a[3] in [8, 9]:
    print("ignore")
else:
    print("answer")
