a = [int(input()) for _ in range(int(input()))]

for i in a:
  if i % 2 == 0:
    print(f"{i} is even")
  else:
    print(f"{i} is odd")
