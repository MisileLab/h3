from datetime import time, date, datetime, timedelta
from math import floor
a, b, c, d = map(int, input().split(" "))
duration: timedelta = datetime.combine(date.min, time(c, d)) - datetime.combine(date.min, time(a, b))
_cache = int(duration.seconds / 60)
print(f"{_cache} {floor(_cache / 30)}")
