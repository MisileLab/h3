a = []

def percent(c: int):
  return c / 100 * 30

for _ in range(int(input())):
  a.append(map(int, input().split(" ")))

for i in a:
  b, c, d, e = i
  if sum([c, d, e]) >= 55 and c >= percent(35) and d >= percent(25) and e >= percent(40):
    print(f"{b} {sum([c, d, e])} PASS")
  else:
    print(f"{b} {sum([c, d, e])} FAIL")
