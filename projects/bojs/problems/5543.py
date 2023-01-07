a = []
b = []

for _ in range(3):
    a.append(int(input()))

for _ in range(2):
    b.append(int(input()))

print(f"{(list(sorted(a))[0] + list(sorted(b))[0]) - 50}")
