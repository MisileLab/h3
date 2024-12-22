from _typeshed import Incomplete
from dataclasses import dataclass

comcigan_url: str
ua: Incomplete

class SchoolNotFound(Exception): ...
class SchoolDuplicated(Exception): ...

class Day:
    monday: int
    tuesday: int
    wednesday: int
    thursday: int
    friday: int

@dataclass
class Lecture:
    period: int
    subject: str
    teacher: str
    def __init__(self, period, subject, teacher) -> None: ...

@dataclass
class TimeTableData:
    period: int
    subject: str
    teacher: str
    replaced: bool
    original: Lecture | None
    def __init__(self, period, subject, teacher, replaced, original) -> None: ...

@dataclass
class TimeTable:
    school_code: int
    school_name: str
    local_code: int
    local_name: str
    school_year: int
    start_date: str
    day_time: list[str]
    update_date: str
    timetable: dict[int, dict[int, dict[int, list[TimeTableData]]]]
    homeroom_teacher: list[list[str]]
    def homeroom(self, grade: int, cls: int) -> str: ...
    def __init__(self, school_code, school_name, local_code, local_name, school_year, start_date, day_time, update_date, timetable, homeroom_teacher) -> None: ...

def fetch_timetable(school_name: str, local_code: int = 0, school_code: int = 0, next_week: bool = False) -> TimeTable: ...
def get_code() -> tuple[str, str, str, str, str, str, str]: ...
def get_school_code(school_name: str, local_code: int, school_code: int, comcigan_code: str) -> tuple[int, str, int]: ...
