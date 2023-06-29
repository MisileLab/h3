from fastapi import FastAPI, HTTPException
from bs4 import BeautifulSoup

import requests
import ssl

url = 'https://www.tauron-dystrybucja.pl/'

class TLSAdapter(requests.adapters.HTTPAdapter):

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.set_ciphers('DEFAULT@SECLEVEL=1')
        kwargs['ssl_context'] = ctx
        return super(TLSAdapter, self).init_poolmanager(*args, **kwargs)

session = requests.session()
session.mount('https://', TLSAdapter())

app = FastAPI()
mapping = {
    "급식": "_meal",
    "등록일": "date",
    "제목": "title",
    "식단": "meal",
    "칼로리": "kcal"
}

@app.get("/{date}")
def get_meal(date: int):
    a = session.post("https://sunrint.sen.hs.kr/dggb/module/mlsv/selectMlsvDetailPopup.do", data={"mlsvId": date})
    b = {
        "_meal": "error",
        "date": "date",
        "title": "title",
        "meal": "meal",
        "kcal": "kcal"
    }
    if not a.ok:
        raise HTTPException(status_code=a.status_code)
    s = BeautifulSoup(a.content, "lxml")
    for i in s.find_all("tr"):
        b[mapping[i.th.text]] = [x for x in i.td.text.replace("\t", "").split("\r\n") if x != ""]
    try:
        b["kcal"] = int(b["kcal"][0].replace("kcal", ""))
    except ValueError:
        b["kcal"] = None
    print(b["meal"])
    for i3, i in enumerate(b["meal"]):
        print(i)
        org = ""
        for i2 in reversed(i):
            print(i2)
            if not i2.isdigit() and i2 != '.':
                org += i2
            else:
                continue
        b["meal"][i3] = "".join(reversed(org))
    return b
