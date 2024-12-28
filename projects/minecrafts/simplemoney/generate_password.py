from argon2 import PasswordHasher

from string import ascii_letters, digits
from secrets import SystemRandom

for _ in range(int(input())):
  p = "".join(SystemRandom().choices(ascii_letters + digits, k=16))
  print(p)
  print(PasswordHasher().hash(p))
