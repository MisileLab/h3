for i, i2 in enumerate(int(input()) for _ in range(int(input()))):
  a = 'Round 1'
  if i2 <= 25:
    a = 'World Finals'
  elif i2 <= 1000:
    a = 'Round 3'
  elif i2 <= 4500:
    a = 'Round 2'
  print(f"Case #{i+1}: {a}")

