# use code in https://gogetem.tistory.com/entry/%EB%B0%B1%EC%A4%80-8710%EB%B2%88-Koszykarz-Python3

a, b, c = map(int, input().split(" "))
d = (b-a)//c
if (a-b) % c:
  print(d+1)
else:
  print(d)
