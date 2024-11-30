class Base:
  def __init__(self):
    self.lose = None
    self.win = None

  def compare(self, a):
    if isinstance(a, self.lose):
      return -1
    elif isinstance(a, self.win):
      return 1
    else:
      return 0

class Rock(Base):
  def __init__(self):
    self.win = Scissor
    self.lose = Paper

class Scissor(Base):
  def __init__(self):
    self.win = Paper
    self.lose = Rock

class Paper(Base):
  def __init__(self):
    self.win = Rock
    self.lose = Scissor

def string_to_class(x: str):
  if x == 'R':
    return Rock()
  elif x == 'S':
    return Scissor()
  else:
    return Paper()

for _ in range(int(input())):
  p, p2 = 0, 0
  for _ in range(int(input())):
    x, y = map(str, input().split(" "))
    res = string_to_class(x).compare(string_to_class(y))
    if res == 1:
      p += 1
    elif res == -1:
      p2 += 1
  if p > p2:
    print("Player 1")
  elif p == p2:
    print("TIE")
  else:
    print("Player 2")
