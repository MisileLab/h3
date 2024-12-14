from fastapi import APIRouter, HTTPException, Query, status
from pydantic import Field

from os import environ
from dataclasses import dataclass
from datetime import datetime
from re import sub

from libraries.request import get

app = APIRouter(prefix="/lunch/{school}")

@dataclass
class LunchData:
  date: str = Field(description="date of day", examples=["20240604"])
  menu: list[str] = Field(description="menu of day")

@dataclass
class LunchDatas:
  menus: list[LunchData] = Field(description="menus of data", default=[])

@dataclass
class Range:
  start: int
  end: int

@app.get("/", description="get school's lunch, total days must less than 31")
async def lunch(
  school: str,
  year: Range = Query(description="year range of lunch"),
  month: Range = Query(description="month range of lunch"),
  day: Range = Query(description="day range of lunch")
) -> LunchDatas:
  l = LunchDatas()
  if (datetime(year.start, month.start, day.start) - datetime(year.end, month.end, day.end)).days >= 30:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="total days must less than 31")
  data = get("https://open.neis.go.kr/hub/mealServiceDietInfo", params={
    "key": environ["NEIS_API"],
    "type": "json",
    "ATPT_OFCDC_SC_CODE": "B10", # change
    "SD_SCHUL_CODE": 7010536, # change
    "MLSV_FROM_YMD": f"{year.start}{month.start:2d}{day.start:2d}",
    "MLSV_TO_YMD": f"{year.start}{month.start:2d}{day.start:2d}"
  })
  for row in data['mealServiceDietInfo'][1]['row']:
    menu = row['DDISH_NM']
    lunch_data = LunchData(
      date=row['MLSV_YMD'],
      menu=[sub(r'\s*\([^)]*\)', '', item.strip()) for item in menu.split('<br/>')]
    )
    l.menus.append(lunch_data)
  return l
