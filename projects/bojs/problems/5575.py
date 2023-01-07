from datetime import datetime, timedelta

a = []

for _ in range(3):
    a.append(input().split(" "))

def days_hours_minutes(td: timedelta):
    return [str(td.seconds//3600), str((td.seconds//60)%60), str((td.seconds%3600)%60)]

def b(c: list):
    s1 = ' '.join([c[0], c[1], c[2]])
    s2 = ' '.join([c[3], c[4], c[5]])
    format = '%H %M %S'

    time = datetime.strptime(s2, format) - datetime.strptime(s1, format)
    return ' '.join(days_hours_minutes(time))

for i in a:
    print(b(i))
    
