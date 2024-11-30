class Timer:
  def __init__(self, hs: map):
    h, m, s = hs
    self.h = h
    self.m = m
    self.s = s

  def add_second(self, s: int):
    if s >= 60:
      m = (s - s % 60) / 60
      s %= 60
      a = self.add_minute(m)
      if a is not None and a <= 24:
        self.s = 0
      self.add_second(s)
    elif s + self.s >= 60:
      self.add_minute(1)
      self.s = (s + self.s) - 60
    else:
      self.s += s
  
  def add_minute(self, m: int):
    if m >= 60:
      return self._extracted_from_add_minute_3(m)
    elif m + self.m >= 60:
      self.add_hour(1)
      self.m = (m + self.m) - 60
    else:
      self.m += m
    return None

  def _extracted_from_add_minute_3(self, m):
    h = (m - m % 60) / 60
    if h >= 24:
      self.m = 0
    m %= 60
    self.add_hour(h)
    self.add_minute(m)
    return h

  def add_hour(self, h: int):
    if self.h + h >= 24:
      h = (self.h + h) % 24
      self.h = 0
    self.h += h

  def add_time(self, s: int):
    if s >= 3600:
      self.add_hour((s - s % 3600) / 3600)
      s %= 3600
    if s >= 60:
      self.add_minute((s - s % 60) / 60)
      s %= 60
    self.add_second(s)
    self.h = int(self.h)
    self.m = int(self.m)
    self.s = int(self.s)
    return " ".join([str(self.h), str(self.m), str(self.s)])

print(Timer(map(int, input().split(" "))).add_time(int(input())))
