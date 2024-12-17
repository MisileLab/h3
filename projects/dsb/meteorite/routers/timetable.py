from fastapi import APIRouter
from pydantic import Field
from pcfrv.timetable import TimeTableData, fetch_timetable

from dataclasses import dataclass
from typing import Optional

app = APIRouter()

@dataclass
class BaseClass:
  period: int
  subject: str
  teacher: str

@dataclass
class ClassInfo(BaseClass):
  replaced: bool
  original: Optional[BaseClass] = None

@app.get("/timetable")
async def get_timetable(
  name: str = Field(description="name of school, can be autocompleted"),
  grade: int = Field(description="grade of timetable"),
  s_class: int = Field(description="class of timetable", alias="class"),
  next_week: bool = Field(description="if True, returns next week's timetable else, returns current week's timetable", default=False)
) -> TimeTableData:
  return fetch_timetable(name, next_week=next_week).timetable[grade][s_class][1]
