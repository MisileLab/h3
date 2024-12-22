from httpx import get
from fake_useragent import UserAgent
from pydantic import validate_call

from base64 import b64encode
from typing import final
from re import findall
from json import loads
from urllib import parse
from dataclasses import dataclass

comcigan_url = 'http://comci.net:4082'
ua = UserAgent(platforms=["desktop"])

class SchoolNotFound(Exception):
  pass

class SchoolDuplicated(Exception):
  pass

@final
class Day:
  monday = 1
  tuesday = 2
  wednesday = 3
  thursday = 4
  friday = 5

@dataclass
class Lecture:
  period: int
  subject: str
  teacher: str

@dataclass
class TimeTableData:
  period: int
  subject: str
  teacher: str
  replaced: bool
  original: Lecture | None

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

  @validate_call(validate_return=True)
  def homeroom(self, grade: int, cls: int) -> str:
    return self.homeroom_teacher[grade - 1][cls - 1]

@validate_call(validate_return=True)
def fetch_timetable(
  school_name: str,
  local_code: int = 0,
  school_code: int = 0,
  next_week: bool = False
) -> TimeTable:
  comcigan_code, code0, code1, code2, code3, code4, code5 = get_code()
  local_code, school_name, school_code = get_school_code(school_name, local_code, school_code, comcigan_code)
  if local_code == -1:
    raise SchoolDuplicated('학교가 2개 이상 존재합니다.')
  elif local_code == -2:
    raise SchoolNotFound('학교를 찾을 수 없습니다.')
  sc = b64encode(f"{str(code0)}_{school_code}_0_{str(int(next_week) + 1)}".encode('utf-8'))
  resp = get(f'{comcigan_url}{comcigan_code[:7]}{str(sc)[2:-1]}', headers={'User-Agent': ua.random})
  resp.encoding = 'UTF-8'
  resp = loads(resp.text.split('\n')[0])

  t = TimeTable(
    school_code,
    school_name,
    local_code,
    resp["지역명"],
    resp["학년도"],
    resp["시작일"],
    resp["일과시간"],
    resp[f"자료{code3}"],
    {},
    []
  )

  data: dict[int, dict[int, dict[int, list[TimeTableData]]]] = {}
  teacher_list = resp[f"자료{code1}"]
  teacher_list[0] = ""
  sub_list = resp[f"자료{code2}"]
  sub_list[0] = ""

  original_timetable = resp["자료" + code5]
  grade = 1
  for i in resp[f"자료{code4}"][1:]:
    cls = 1
    data[grade] = {}
    for j in i[1:]:
      data[grade][cls] = {}
      for day in range(1, original_timetable[grade][cls][0] + 1):
        data[grade][cls][day] = []
        for period in range(1, original_timetable[grade][cls][day][0] + 1):
          original_period = original_timetable[grade][cls][day][period]
          period_num = 0 if j[day][0] < period else j[day][period]
          data[grade][cls][day].append(TimeTableData(
            period=period,
            subject=sub_list[period_num // 1000],
            teacher=teacher_list[period_num % 100],
            replaced=period_num != original_period,
            original=None if period_num == original_period else Lecture(
              period=period,
              subject=sub_list[original_period // 1000],
              teacher=teacher_list[original_period % 100]
            )
          ))
      cls += 1
    grade += 1

  t.timetable = data
  homeroom_teacher = resp["담임"]
  for grade in range(len(homeroom_teacher)):
    for cls in range(len(homeroom_teacher[grade])):
      if homeroom_teacher[grade][cls] in [0,255]:
        del homeroom_teacher[grade][cls:]
        break
      else:
        homeroom_teacher[grade][cls] = teacher_list[homeroom_teacher[grade][cls]]
  t.homeroom_teacher = homeroom_teacher
  return t

@validate_call(validate_return=True)
def get_code() -> tuple[str, str, str, str, str, str, str]:
  resp = get(f"{comcigan_url}/st", headers={'User-Agent': ua.random})
  resp.encoding = 'euc-kr'
  resp = resp.text
  comcigan_code = findall('\\.\\/[0-9]+\\?[0-9]+l', resp)[0][1:]
  code0 = findall('sc_data\\(\'[0-9]+_', resp)[0][9:-1]
  code1 = findall('성명=자료.자료[0-9]+', resp)[0][8:]
  code2 = findall('자료.자료[0-9]+\\[sb\\]', resp)[0][5:-4]
  code3 = findall('=H시간표.자료[0-9]+', resp)[0][8:]
  code4 = findall('일일자료=Q자료\\(자료\\.자료[0-9]+', resp)[0][14:]
  code5 = findall('원자료=Q자료\\(자료\\.자료[0-9]+', resp)[0][13:]
  return comcigan_code, code0, code1, code2, code3, code4, code5

@validate_call(validate_return=True)
def get_school_code(school_name: str, local_code: int, school_code: int, comcigan_code: str) -> tuple[int, str, int]:
  resp = get(f"{comcigan_url}{comcigan_code}{parse.quote(school_name, encoding='euc-kr')}")
  resp.encoding = 'UTF-8'
  resp = loads(resp.text.strip(chr(0)))
  if len(resp["학교검색"]) == 0:
    return -2, '', -2
  elif len(resp["학교검색"]) > 1:  # 2개 이상이 검색될
    if school_code:
      for data in resp["학교검색"]:
        if data[3] == school_code:
          return data[0], data[2], data[3]
    if local_code:
      for data in resp["학교검색"]:
        if data[0] == local_code:
          return data[0], data[2], data[3]
    return -1, '', -1
  return resp['학교검색'][0][0], resp['학교검색'][0][2], resp['학교검색'][0][3]
