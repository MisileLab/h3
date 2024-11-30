a = [1]
n = int(input())
i = 2
np = 1
while n != i:
  a.append(i)
  if np == 2:
    i -= 1
  else:
    i += 1 if i+2 <= n and (i+1 not in a and np != 1) else 2
  np += 1
  if np == 3:
    np = 0
a.extend([5, 1])
print(n)
print(*a)
