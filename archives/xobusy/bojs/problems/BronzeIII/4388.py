def carrynum(a, b):
  def f(n):
    return sum(map(int, str(n)))
  return(f(a)+f(b)-f(a+b))/9

while True:
  a, b = map(int, input().split(" "))
  if a == 0 and b == 0:
    break
  print(int(carrynum(a, b)))
