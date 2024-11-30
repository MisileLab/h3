a, b, c = map(int, input().split(" "))
d, e, f = map(int, input().split(" "))

def is_greater_but_list(g: list, f: list):
  return all(i >= i2 for i, i2 in zip(f, g))

if is_greater_but_list([a, b, c], [d, e, f]):
  print("A")
elif is_greater_but_list([b, c], [e, f]):
  if d >= a / 2:
    print("B")
  else:
    print("C")
elif e >= b / 2:
  print("D")
else:
  print("E")
