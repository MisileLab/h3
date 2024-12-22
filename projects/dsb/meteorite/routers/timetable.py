from fastapi import APIRouter
from pydantic import Field
from pcfrv.timetable import TimeTableData, fetch_timetable

from dataclasses import dataclass
from typing import Annotated

app = APIRouter()

@dataclass
class BaseClass:
  period: int
  subject: str
  teacher: str

@dataclass
class ClassInfo(BaseClass):
  replaced: bool
  original: BaseClass | None = None

@app.get("/timetable", description="returns dict, key is range of 1(monday) to 5(friday), value is list of classes")
async def get_timetable(
  name: Annotated[str, Field(description="name of school, can be autocompleted")],
  grade: Annotated[int, Field(description="grade of timetable")],
  s_class: Annotated[int, Field(description="class of timetable", alias="class")],
  next_week: Annotated[bool, Field(description="if True, returns next week's timetable else, returns current week's timetable")] = False
) -> dict[int, list[TimeTableData]]:
  return fetch_timetable(name, next_week=next_week).timetable[grade][s_class]
