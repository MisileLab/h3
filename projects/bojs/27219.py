from math import floor

a = int(input())
b = "V" * floor(a / 5)
a -= floor(a / 5) * 5

if a == 4:
    b += "IV"
    a -= 4
else:
    b += "I" * a

print(b)
