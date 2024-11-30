a = input().count('1')
b = input().count('1')

if a >= b or (a + 2) % 2 == 1 and (a+1) == b:
  print("VICTORY")
else:
  print("DEFEAT")