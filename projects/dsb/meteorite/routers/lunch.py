from fastapi import APIRouter, HTTPException, Query, status
from edgedb import create_async_client # pyright: ignore[reportUnknownVariableType]
from pydantic import Field, BaseModel

from os import environ
from datetime import date as pdate
from re import sub
from typing import Annotated

from libraries.request import get

from queries.get_school_async_edgeql import get_school
from queries.add_school_async_edgeql import add_school

app = APIRouter(prefix="/lunch/{school}")
db = create_async_client()

class LunchData(BaseModel):
  date: pdate = Field(description="date of day")
  menu: list[str] = Field(description="menu of day")

class LunchDatas(BaseModel):
  menus: list[LunchData] = Field(description="menus of data", default=[])

@app.get("/", description="get school's lunch, total days must less than 31")
async def lunch(
  school: str,
  start: Annotated[pdate, Query(description="start date of lunch")],
  end: Annotated[pdate, Query(description="end date of lunch")]
) -> LunchDatas:
  if (end - start).days > 30:
    raise HTTPException(
      status_code=status.HTTP_400_BAD_REQUEST,
      detail="total days must less than 31"
    )
  lunch = LunchDatas()
  sel = await get_school(db, name=school)
  if sel is None:
    r = get("https://open.neis.go.kr/hub/schoolInfo", params={
      "key": environ["NEIS_API"],
      "type": "json",
      "SCHUL_NM": school,
      "pSize": 1
    })['schoolInfo'][1]['row']
    if len(r) == 0:
      raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="school doesn't exists"
      )
    r = r[0]
    sel = await get_school(db, name=r['SCHUL_NM'])
    if sel is None:
      _ = await add_school(db, name=r["SCHUL_NM"], school_code=int(r['SD_SCHUL_CODE']), ofcdc_code=r['ATPT_OFCDC_SC_CODE'])
      sel = await get_school(db, name=r["SCHUL_NM"])
      if sel is None:
        print("idk something wrong")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="idk something wrong")
  data = get("https://open.neis.go.kr/hub/mealServiceDietInfo", params={
    "key": environ["NEIS_API"],
    "type": "json",
    "ATPT_OFCDC_SC_CODE": sel.ofcdc_code,
    "SD_SCHUL_CODE": sel.school_code,
    "MLSV_FROM_YMD": f"{start.year}{start.month:02d}{start.day:02d}",
    "MLSV_TO_YMD": f"{end.year}{end.month:02d}{end.day:02d}"
  })
  if data.get('RESULT', {}).get('MESSAGE') == '해당하는 데이터가 없습니다.':
    return lunch
  for row in data['mealServiceDietInfo'][1]['row']:
    menu = row['DDISH_NM']
    date_data = row["MLSV_YMD"]
    lunch_data = LunchData(
      date=pdate(int(date_data[0:4]), int(date_data[4:6]), int(date_data[6:8])),
      menu=[
        sub(r'\s*\([^)]*\)', '', item.strip())
        for item in menu.split('<br/>')
      ]
    )
    lunch.menus.append(lunch_data)
  return lunch
