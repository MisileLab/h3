from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, HTTPException, status
from pcfrv.timetable import SchoolDuplicated, SchoolNotFound, TimeTableData, fetch_timetable
from pcfrv.search_school import School, get_school_code
from pydantic import Field

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

@dataclass
class Classes:
  grade: int
  classes: list[int]

@app.get("/timetable", description="returns dict, key is range of 1(monday) to 5(friday), value is list of classes")
async def get_timetable(
  school: Annotated[str, Field(description="name of school, can be autocompleted")],
  grade: Annotated[int, Field(description="grade of timetable")],
  s_class: Annotated[int, Field(description="class of timetable", alias="class")],
  next_week: Annotated[bool, Field(description="if True, returns next week's timetable else, returns current week's timetable")] = False
) -> dict[int, list[TimeTableData]]:
  try:
    return fetch_timetable(school, next_week=next_week).timetable[grade][s_class]
  except SchoolNotFound:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="School not found")
  except SchoolDuplicated:
    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="School name is duplicated")

@app.get("/search", description="return list of school's name that matches")
async def search(school: Annotated[str, Field(description="name of school")]) -> list[School]:
  try:
    return get_school_code(school)
  except SchoolDuplicated:
    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="School name is duplicated")

@app.get("/classlist", description="return list of classes")
async def classlist(school: Annotated[str, Field(description="name of school")]) -> list[Classes]:
  try:
    t = fetch_timetable(school)
    res: list[Classes] = []
    for g, v in t.timetable.items():
      res.append(Classes(g, []))
      for c in v.keys():
        res[-1].classes.append(c)
    return res
  except SchoolNotFound:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="School not found")
  except SchoolDuplicated:
    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="School name is duplicated")

