a = int(input())

if a == 0:
  print("divide by zero")
else:
  b = list(map(int, input().split(" ")))
  print("{:.2f}".format((sum(b)/a) / (sum(i*(b.count(i)/a) for i in list(dict.fromkeys(b))))))
