newline = "\n"

class Architectures:
  little_endian = 1

def string_to_hexcode(n: str):
  return ''.join(['{:02x}'.format(ord(c)) for c in n[::-1]])

def div_string(n: str, a: int):
  b = []
  c = ''
  for i, i2 in enumerate(n):
  c += i2
  if (i+1) % a == 0:
    b.append(c)
    c = ''
  if c != '':
  b.append(c)
  return b
  

def hexcode_to_asm(n: str, arch: Architectures = Architectures.little_endian, reg: str = "rax"):
  if arch == Architectures.little_endian:
  strs = div_string(n, 16)
  return f"""push 0x0
{newline.join(f'mov {reg}, 0x{i}{newline}push {reg}' for i in strs)}"""

