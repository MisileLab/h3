from embed import ascii_range
from numpy import array, uint8

f = ""
byte1 = bytes.fromhex("AC F3 0C 25 A3 10 B7 25 16 C6 B7 BC 07 25 02 D5 C6 11 07 C5 00 00 00 00 00 00 00 00 00 00 00 00")

for i in range(21):
  for i2 in ascii_range():
    if array(i2 * -5).astype(uint8) == byte1[i]:
      f += chr(i2)
      break
  else:
    print("no flag")
    break

print(f)
