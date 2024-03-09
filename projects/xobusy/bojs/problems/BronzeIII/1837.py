a, b = map(int, input().split(" "))

for i in range(2, b+1):
  if a % i == 0 and i < b:
    print(f"BAD {i}")
    break
else:
  print("GOOD")
