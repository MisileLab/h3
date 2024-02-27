class AssemblyBuilder:
  def __init__(self):
    self.asmlist = []

  def retn(self) -> str:
    return '\n'.join(asmlist)

  """init_reg"""
  def ireg(self, r: str, v: int = 0):
    val = f"mov {r} {v}"
    if v == 0:
      val = f"xor {r} {r}"
    return val

  def mov(self, r1: str, r2: str):
    self.asmlist.append(f"mov {r1} {r2}")

  def xor(self, r1: str, r2: str):
    self.asmlist.append(f"xor {r1} {r2}")

  def sub(self, r1: str, r2: str):
    self.asmlist.append(f"sub {r1} {r2}")
