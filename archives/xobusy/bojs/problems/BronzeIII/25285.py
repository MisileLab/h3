a = []

def return_number(n: float, n2: float):
  if n < 140.1:
    return 6
  elif n < 146.0:
    return 5
  elif n < 159.0:
    return 4
  elif n < 161.0:
    return 3 if n2 >= 16.0 and n2 < 35.0 else 4
  elif n < 204.0:
    if n2 < 25.0 and n2 >= 20.0:
      return 1
    elif (n2 >= 18.5 and n2 < 20.0) or (n2 >= 25.0 and n2 < 30.0):
      return 2
    elif (n2 >= 16.0 and n2 < 18.5) or (n2 >= 30.0 and n2 < 35.0):
      return 3
    else:
      return 4
  else:
    return 4

for _ in range(int(input())):
  b, c = map(float, input().split(" "))
  d = b / 100
  a.append(return_number(b, c/(d*d)))

for i in a:
  print(i)
