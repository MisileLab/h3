for i in [map(float, input().split(" ")) for _ in range(int(input()))]:
  a, b = i
  h = (a/b)*2
  print(f"The height of the triangle is {h:.2f} units")
