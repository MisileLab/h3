for i in [list(map(int, input().split())) for _ in range(int(input()))]:
  if i[0] * i[1] == i[2] * i[3]:
    print('Tie')
  elif i[0] * i[1] < i[2] * i[3]:
    print('Eurecom')
  else:
    print('TelecomParisTech')

