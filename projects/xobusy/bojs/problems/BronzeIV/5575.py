from datetime import datetime, timedelta

a = [input().split(" ") for _ in range(3)]

def days_hours_minutes(td: timedelta):
  return [str(td.seconds//3600), str((td.seconds//60)%60), str((td.seconds%3600)%60)]

def b(c: list):
  return ' '.join(days_hours_minutes(datetime.strptime(' '.join([c[3], c[4], c[5]]), '%H %M %S') - datetime.strptime(' '.join([c[0], c[1], c[2]]), '%H %M %S')))

for i in a:
  print(b(i))
