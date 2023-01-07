from datetime import timedelta, datetime

s1 = '11 11 11'
s2 = ' '.join(input().split(" "))
format = '%d %H %M'

a = int((datetime.strptime(s2, format) - datetime.strptime(s1, format)).total_seconds()//60)
if a < 0:
    print(-1)
else:
    print(a)

