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

def _meal_backend(year: int, month: int):
    if month < 10:
        month = f"0{month}"
    r = session.post(f"https://school.koreacharts.com/school/meals/B000011299/{year}{month}.html")
    if not r.ok:
        raise HTTPException(status_code=r.status_code)
    b = BeautifulSoup(r.content, "lxml")
    c = b.find_all("tbody")[1].find_all("tr")[1:]
    meallist = []
    for i in c:
        _elementlist = i.find_all("td")
        meallist.append({
            "meal": [x for x in _elementlist[2].text.replace("\t", "").replace(" ", "").split("\n") if x != ""][1:],  # noqa: E501
            "weekday": _elementlist[1].text,
            "day": int(_elementlist[0].text)
        })
    for i in meallist:
        for i4, i2 in enumerate(i["meal"]):
            if i2.count(".") > 0:
                i["meal"][i4] = "".join(i3 for i3 in i2 if not i3.isdigit() and i3 not in ["."])[:-2]  # noqa: E501
    return meallist

def meal_backend(year: int, month: int):
    try:
        return _meal_backend(year, month)
    except IndexError:
        return []

@app.get("/{year}/{month}/{day}")
def get_meal(year: int, month: int, day: int):
    if a := [x for x in meal_backend(year, month) if x["day"] == day]:
        return a
    else:
        raise HTTPException(status_code=404)

@app.get("/{year}/{month}")
def get_meal_month(year: int, month: int):
    if a := meal_backend(year, month):
        return a
    else:
        raise HTTPException(status_code=404)
