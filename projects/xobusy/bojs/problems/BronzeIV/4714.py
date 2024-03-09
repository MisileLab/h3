a = []

while True:
  b = float(input())
  if b < 0:
    break
  a.append(b)

for i in a:
  print(f"Objects weighing {float(i):.2f} on Earth will weigh {float(i * 0.167):.2f} on the moon.")
