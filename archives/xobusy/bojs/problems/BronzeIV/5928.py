from datetime import datetime

a = int((datetime.strptime(' '.join(input().split(" ")), "%d %H %M") - datetime.strptime('11 11 11', "%d %H %M")).total_seconds()//60)
if a < 0:
  print(-1)
else:
  print(a)
