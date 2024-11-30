for i in [map(float, input().split(" ")) for _ in range(int(input()))]:
  a, b = i
  print(f"{max(a, b) - min(a, b):.1f}")
