from httpx import get
from fake_useragent import UserAgent
from pydantic import validate_call

from re import findall
from json import loads
from urllib import parse
from dataclasses import dataclass

@dataclass
class School:
  name: str
  period: str
  school_code: int
  period_code: str

comcigan_url = 'http://comci.net:4082'

@validate_call(validate_return=True)
def get_code() -> str:
  resp = get(f"{comcigan_url}/st", headers={"User-Agent": UserAgent(platforms=["desktop"]).random})
  resp.encoding = 'euc-kr'
  return findall('\\.\\/[0-9]+\\?[0-9]+l', resp.text)[0][1:]

@validate_call(validate_return=True)
def get_school_code(school_name: str) -> list[School]:
  resp = get(f"{comcigan_url}{get_code()}{parse.quote(school_name, encoding='euc-kr')}")
  resp.encoding = 'UTF-8'
  return [School(i[2],i[1],i[0],i[3]) for i in loads(resp.text.strip(chr(0)))["학교검색"]]
