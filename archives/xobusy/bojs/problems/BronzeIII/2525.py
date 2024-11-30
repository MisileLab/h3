a, b = map(int, input().split(" "))
c = int(input())
a += c // 60
b += c % 60
if b >= 60:
  b -= 60
  a += 1
if a >= 24:
  a -= 24
print(a, b)
