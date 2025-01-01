from numpy import astype, int8, array

data = bytes.fromhex("49 60 67 74 63 67 42 66 80 78 69 69 7B 99 6D 88 68 94 9F 8D 4D A5 9D 45 00 00 00 00 00 00 00 00")
res: list[int8] = []

for i in range(0x18):
  t = data[i] - 2 * i
  _ = res.append(astype(array([i]), int8)[0] ^ t) # pyright: ignore[reportAny]

print(bytes(res))
