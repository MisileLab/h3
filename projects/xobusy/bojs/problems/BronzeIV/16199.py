a1, a2, a3 = map(int, input().split(" "))
b1, b2, b3 = map(int, input().split(" "))

a = b1 - a1
b = a

if b2 - a2 < 0 or (b2 - a2 <= 0 and b3 - a3 < 0):
  a -= 1

print(a)
print(b+1)
print(b)

