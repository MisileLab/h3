from math import floor
a = floor(int(input()) / 5)
b = int(input())

def discount(price: int, n: int):
  return price-n if price >= n else 0

class Dummy:
  def __init__(self, price: int):
    self.final = price

  def set_price(self, price: int):
    self.final = min(self.final, price)

d = Dummy(b)
if a >= 1:
  d.set_price(discount(b, 500))
if a >= 2:
  d.set_price(b-b // 10)
if a >= 3:
  d.set_price(discount(b, 2000))
if a >= 4:
  d.set_price(b-b // 4)

print(d.final)
