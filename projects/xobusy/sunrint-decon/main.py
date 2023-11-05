from typing import Union
from pathlib import Path
import base64

from fastapi import FastAPI, Header, status, HTTPException
from pymongo import MongoClient
from tomli import loads
from requests import get

config = loads(Path("config.toml").read_text())
app = FastAPI()
db = MongoClient(config["MONGO_URI"])["schoolfinder"]

def is_valid(name: str, password: str) -> bool:
    return db["login"].find_one({"name":name,"password":password}) != None

@app.post("/register")
async def register(
    name: Union[str, None] = Header(default=None),
    password: Union[str, None] = Header(default=None)
) -> None:
    if name is None or password is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

    if db["login"].find_one({"name":name}):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT)

    db["login"].insert_one({"name":name,"password":password})

@app.get("/school/search")
async def school_search(name: Union[str, None] = Header(default=None)) -> list[dict]:
    if name is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
    name = base64.b64decode(name).decode("utf-8")
    a = get("https://open.neis.go.kr/hub/schoolInfo",params={"KEY":config["NEIS_KEY"],"TYPE":"json","SCHUL_NM":name,"SCHUL_KND_SC_NM":"고등학교"})
    a.raise_for_status()
    return [{
        "name": i["SCHUL_NM"],
        "kind": i["HS_SC_NM"],
        "code": i["SD_SCHUL_CODE"],
        "homepage": i["HMPG_ADRES"],
        "address": i["ORG_RDNMA"],
        "callnumber": i["ORG_TELNO"],
        "created": i["FOND_YMD"]
    } for i in a.json()["schoolInfo"][1]["row"]]

@app.get("/school/review")
async def school_review(code: int) -> list[dict]:
    return list(db["info"].find({"code":code}))[0]["reviews"]

@app.post("/school/review")
async def school_review_post(
    code: Union[int, None] = Header(default=None),
    username: Union[str, None] = Header(default=None),
    password: Union[str, None] = Header(default=None),
    review: Union[str, None] = Header(default=None),
    stars: Union[float, None] = Header(default=None)
):
    if code is None or username is None or password is None or review is None or stars is None or stars > 5 or stars < 0.5 or stars % 0.5 != 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

    if not is_valid(username,password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    data = db["info"].find({"code":code})
    if data is None:
        db["info"].insert_one({"code":code,"reviews":[]})
        data = db["info"].find({"code":code})

    data = list(data)[0]

    if [i for i in data['reviews'] if i["username"] == username]:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT)

    data["reviews"].append({"username":username,"review":review,"star":stars})
    db["info"].update_one({"code":code},{"$set":{"reviews":data["reviews"]}})

