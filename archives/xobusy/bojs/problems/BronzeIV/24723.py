def notfactorial(a: int):
  return 2 if a == 1 else notfactorial(a-1) * 2

print(notfactorial(int(input())))
