a = [list(map(int, input().split(" "))) for _ in range(int(input()) * 2)]

for i in range(0, len(a), 2):
  b, c = a[i], a[i+1]
  _s = b[0] + b[1] * 2 + (b[2] + b[3]) * 3 + b[4] * 4 + b[5] * 10
  _s2 = c[0] + (c[1] + c[2] + c[3]) * 2 + c[4] * 3 + c[5] * 5 + c[6] * 10
  strs = f"Battle {i//2+1}:"
  if _s > _s2:
    print(strs, "Good triumphs over Evil")
  elif _s < _s2:
    print(strs, "Evil eradicates all trace of Good")
  else:
    print(strs, "No victor on this battle field")
