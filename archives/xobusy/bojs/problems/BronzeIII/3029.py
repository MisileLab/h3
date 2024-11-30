from datetime import datetime
from string import Template

class DeltaTemplate(Template):
  delimiter = "%"

def strfdelta(tdelta, fmt):
  d = {"D": tdelta.days}
  d["H"], rem = divmod(tdelta.seconds, 3600)
  d["M"], d["S"] = divmod(rem, 60)
  d["H"] = str(d["H"])
  d["M"] = str(d["M"])
  d["S"] = str(d["S"])
  if len(d["H"]) == 1:
    d["H"] = "0" + d["H"]
  if len(d["M"]) == 1:
    d["M"] = "0" + d["M"]
  if len(d["S"]) == 1:
    d["S"] = "0" + d["S"]
  t = DeltaTemplate(fmt)
  return t.substitute(**d)

a = _tmp = datetime.strptime(input(), "%H:%M:%S")
b = datetime.strptime(input(), "%H:%M:%S")
if a.hour + (b.hour - a.hour if b > a else a.hour - b.hour) + (24 if a.hour == b.hour else 0) >= 24:
  a = a.replace(day=2 if b > a else 1)
  b = b.replace(day=2 if _tmp > b else 1)
if a == b:
  print("24:00:00")
else:
  print(strfdelta(b - a, "%H:%M:%S"))
