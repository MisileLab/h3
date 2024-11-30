from embed import ascii_range
from numpy import uint8

def rol(data, shift, size=32):
  shift %= size
  remains = data >> (size - shift)
  body = (data << shift) - (remains << size )
  return (body + remains)

byte = bytes.fromhex("52 DF B3 60 F1 8B 1C B5 57 D1 9F 38 4B 29 D9 26 7F C9 A3 E9 53 18 4F B8 6A CB 87 58 5B 39 1E 00")
flag = ""

for i in range(31):
  for i2 in ascii_range():
    if i ^ uint8(rol(i2, i & 7, 8)) == byte[i]:
      flag += chr(i2)
      break
  else:
    print("no flag")

print(flag)
