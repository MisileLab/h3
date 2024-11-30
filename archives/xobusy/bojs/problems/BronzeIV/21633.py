a = int(input())
b = 100
if 25+a / 100 > b:
  b = 25+a / 100
print(f"{min(b, 2000):.2f}")
