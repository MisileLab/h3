from math import floor

a = int(input())
b = "V" * floor(a / 5)
a -= floor(a / 5) * 5
b += "I" * a

print(b)
