from sys import stdout

def printr(a):
  stdout.write(str(a))

n = int(input())
i = 2

printr("3 ")
_n = n - 1
while True:
  _n -= 1
  if _n <= 0:
    break
  printr(f"{i} ")
  i += 2

if n > 1:
  printr(f"{i + (4 if n % 3 == 2 else 0)}")
