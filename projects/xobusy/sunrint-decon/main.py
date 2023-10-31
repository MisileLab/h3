from typing import Union
from pathlib import Path

from fastapi import FastAPI, Header, status, HTTPException
from pymongo import MongoClient
from tomli import loads
from requests import get

config = loads(Path("config.toml").read_text())
app = FastAPI()
db = MongoClient(config["MONGO_URI"])["schoolfinder"]["login"]

async def is_valid(name: str, password: str) -> bool:
    return db.find_one({"name":name,"password":password}) != None

@app.post("/register")
async def register(
    name: Union[str, None] = Header(default=None),
    password: Union[str, None] = Header(default=None)
):
    if name is None or password is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

    if db.find_one({"name":name}):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT)

    db.insert_one({"name":name,"password":password})

@app.get("/school/search")
async def school_search(name: str):
    a = get("https://open.neis.go.kr/hub/schoolinfo",params={"KEY":config["NEIS_KEY"],"TYPE":"json","SCHUL_NM":name})
    a.raise_for_status()
    return a.json()["schoolInfo"][2]["row"]

