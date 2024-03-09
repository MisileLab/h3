a = ['D', 'C', 'B', 'A', 'E']

for _ in range(3):
  print(a[list(map(int, input().split(" "))).count(1)])
