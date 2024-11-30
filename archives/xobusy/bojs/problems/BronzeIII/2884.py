from datetime import timedelta

b, c = map(int, input().split())
d = timedelta(hours=b, minutes=c) - timedelta(hours=0, minutes=45)
print(d.seconds // 3600, d.seconds % 3600 // 60)
