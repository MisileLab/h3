from secrets import SystemRandom
from string import ascii_lowercase, ascii_uppercase, digits

entire = ascii_lowercase + ascii_uppercase + digits
print(entire)
r = SystemRandom().randint(1, 63)
print(r)

for _ in range(r):
  print(entire[SystemRandom().randint(0, len(entire)-1)], end='')
