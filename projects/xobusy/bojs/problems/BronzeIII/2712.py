for i in [input().split(" ") for _ in range(int(input()))]:
  _temp = 0
  _temp2 = ""
  calc = i[1]
  amount = float(i[0])
  if calc == "kg":
    _temp = amount * 2.2046
    _temp2 = "lb"
  elif calc == "lb":
    _temp = amount * 0.4536
    _temp2 = "kg"
  elif calc == "l":
    _temp = amount * 0.2642
    _temp2 = "g"
  else:
    _temp = amount * 3.7854
    _temp2 = "l"
  print(f"{_temp:.4f} {_temp2}")
