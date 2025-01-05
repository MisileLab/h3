from numpy import astype, uint8, array

from pathlib import Path

keys = astype(array([0xDE, 0xAD, 0xBE, 0xEF]), uint8)
pngheader = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]

encrypted = Path("./encrypted").read_bytes()
original = bytearray()

v5 = 0
for i in encrypted:
  if v5 == len(pngheader):
    j = 0
    while j < len(pngheader):
      if original[j] != pngheader[j]:
        print("unencrypt failed")
        exit(1)
      j += 1
  v = uint8(i)
  v -= 19
  v ^= keys[v5 % 4]
  v5 += 1
  print(v)
  original.append(v)

_ = Path("./original.png").write_bytes(original)
