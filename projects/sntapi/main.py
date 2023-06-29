from fastapi import FastAPI, HTTPException
from bs4 import BeautifulSoup

import requests
import ssl

DEF = {"_meal": ["\n"], "date": "년 월 일 요일", "title": ["\n"], "meal": ["\n"], "kcal": None}  # noqa: E501

class TLSAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.set_ciphers('DEFAULT@SECLEVEL=1')
        kwargs['ssl_context'] = ctx
        return super(TLSAdapter, self).init_poolmanager(*args, **kwargs)

session = requests.session()
session.mount('https://', TLSAdapter())

app = FastAPI()

def list_replace(a: str, b: list, isdigit: bool):
    for i in b:
        a = a.replace(i, "")
        if isdigit:
            a = "".join([x for x in a if not x.isdigit()])
    print(a)
    return a

def meal_backend(year: int, month: int):
    if month < 10:
        month = f"0{month}"
    r = session.post(f"https://school.koreacharts.com/school/meals/B000011299/{year}{month}.html")
    if not r.ok:
        raise HTTPException(status_code=r.status_code)
    b = BeautifulSoup(r.content, "lxml")
    c = b.find_all("tbody")[1].find_all("tr")
    del c[0]
    meallist = []
    for i in c:
        _elementlist = i.find_all("td")
        print(_elementlist)
        meallist.append({
            "meal": [x for x in list_replace(_elementlist[2].text, [" ", "\t", "(", ")", "."], True).split("\n") if x != ""][1:],  # noqa: E501
            "weekday": _elementlist[1].text,
            "day": int(_elementlist[0].text)
        })
    return meallist

@app.get("/{year}/{month}/{day}")
def get_meal(year: int, month: int, day: int):
    return [x for x in meal_backend(year, month) if x["day"] == day]

@app.get("/{year}/{month}")
def get_meal_month(year: int, month: int):
    return meal_backend(year, month)
