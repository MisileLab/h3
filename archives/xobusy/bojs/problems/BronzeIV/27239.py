from string import ascii_lowercase
from math import ceil

a = int(input())
print(ascii_lowercase[((a % 8) - 1) % 8] + str(ceil(a / 8)))
