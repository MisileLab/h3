from typing import Union
from pathlib import Path

from fastapi import FastAPI, Header, status, HTTPException
from pymongo import MongoClient

app = FastAPI()
db = MongoClient(Path("MONGO_URI").read_text().strip())["local"]["schoolfinder"]

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

