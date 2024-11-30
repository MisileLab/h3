b = int(input())
a = int(input())

if a - b >= 31:
  print("You are speeding and your fine is $500.")
elif a - b >= 21:
  print("You are speeding and your fine is $270.")
elif a - b >= 1:
  print("You are speeding and your fine is $100.")
else:
  print("Congratulations, you are within the speed limit!")
