from os import _exit

a, b = map(int, input().split(" "))

for i in range(2, a+1):
    if a % i == 0 and i < b:
        print(f"BAD {i}")
        _exit(0)
print("GOOD")

