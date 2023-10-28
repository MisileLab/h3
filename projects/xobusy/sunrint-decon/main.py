from typing import Union
from pathlib import Path

from fastapi import FastAPI, Header, status, HTTPException
from pymongo import MongoClient

app = FastAPI()
db = MongoClient(Path("MONGO_URI").read_text().strip())["local"]["schoolfinder"]

async def is_valid(name: str, password: str) -> bool:
    return db.find_one({"name":name,"password":password}) != None

