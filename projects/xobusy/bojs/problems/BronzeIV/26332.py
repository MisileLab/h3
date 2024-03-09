for i in [map(int, input().split(" ")) for _ in range(int(input()))]:
  b, c = i
  print(b, c)
  print(b*c-(b-1)*2)
