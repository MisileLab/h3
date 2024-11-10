from webbrowser import open

a = []
running = True

while running:
  try:
    a.append(input())
  except (EOFError, KeyboardInterrupt):
    running = False

for i in a:
  print(i)
  open(i)

