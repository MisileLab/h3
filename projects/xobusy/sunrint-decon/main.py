from typing import Union
from pathlib import Path

from fastapi import FastAPI, Header, status, HTTPException
from pymongo import MongoClient
from tomli import loads
from requests import get

config = loads(Path("config.toml").read_text())
app = FastAPI()
db = MongoClient(config["MONGO_URI"])["schoolfinder"]

async def is_valid(name: str, password: str) -> bool:
    return db["login"].find_one({"name":name,"password":password}) != None

@app.post("/register")
async def register(
    name: Union[str, None] = Header(default=None),
    password: Union[str, None] = Header(default=None)
):
    if name is None or password is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

    if db["login"].find_one({"name":name}):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT)

    db["login"].insert_one({"name":name,"password":password})

@app.get("/school/search")
async def school_search(name: str):
    a = get("https://open.neis.go.kr/hub/schoolinfo",params={"KEY":config["NEIS_KEY"],"TYPE":"json","SCHUL_NM":name})
    a.raise_for_status()
    return a.json()["schoolInfo"][2]["row"]

@app.get("/school/review")
async def school_review(code: int):
    return db["info"].find_one({"code":code})

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

    data = db["info"].find_one({"code":code})
    if data is None:
        db["info"].insert_one({"code":code,"reviews":[]})
        data = db["info"].find_one({"code":code})

    if [i for i in data if i["username"] == username]:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT)

    data["reviews"].append({"username":username,"review":review,"star":stars})
    db["info"].update_one({"code":code},{"$set":{"reviews":data["reviews"]}})

