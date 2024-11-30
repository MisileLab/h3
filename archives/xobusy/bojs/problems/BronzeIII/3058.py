for i in [[x for x in map(int, input().split()) if x % 2 == 0] for _ in range(int(input()))]:
  print(sum(i), min(i))
