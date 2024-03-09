a = input()
b = len(a)

if a.count(a[0]) == len(a):
  print(-1)
  exit()

def is_palindrome(s):
  left = 0
  right = len(s) - 1

  while left < right:
    if s[left] != s[right]:
      return False
    left += 1
    right -= 1

  return True

for i in range(len(a)):
  if not is_palindrome(a[i:]):
    print(b)
    break
  b -= 1
else:
  print(-1)
