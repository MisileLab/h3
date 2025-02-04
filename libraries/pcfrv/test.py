from pcfrv.timetable import fetch_timetable
from pcfrv.search_school import get_school_code

print(fetch_timetable('선린인').timetable[1][1][1])
print(get_school_code("sans"))
print(get_school_code("선린"))

