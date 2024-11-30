_i, a = map(int, input().split(" "))
b = [int(input()) for _ in range(_i)]
_nump = a // sum(b)
for i in b:
  print(_nump * i)
