a = [input() for _ in range(int(input()))]

for _ in range(int(input())):
  b = int(input())
  _cac = b - 1
  try:
    if _cac < 0:
      raise ValueError
    c = a[_cac]
  except (IndexError, ValueError):
    c = "No such rule"
  print(f"Rule {b}: {c}")
