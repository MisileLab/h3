from fastapi import FastAPI, HTTPException
from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv
from bson.json_util import dumps
from os import environ

load_dotenv()

app = FastAPI()
db = MongoClient(host=environ["HOST"], port=int(environ["PORT"]), username=environ["USERNAME"], password=environ["PASSWORD"]) # noqa: E501

def meal_backend(finder: dict):
    database = db.get_database("local")
    if database.get_collection("sntmeal").count_documents(finder) == 0: # noqa: E501
        return []
    a = []
    for i in list(database.get_collection("sntmeal").find(finder)):
        del i["_id"]
        a.append(i)
    return a

@app.get("/{year}/{month}/{day}")
def get_meal(year: int, month: int, day: int):
    if a := meal_backend({"year": year, "month": month, "day": day}):
        return a
    else:
        raise HTTPException(status_code=404)

@app.get("/{year}/{month}")
def get_meal_month(year: int, month: int):
    if a := meal_backend({"year": year, "month": month}):
        return a
    else:
        raise HTTPException(status_code=404)

@app.get("/{year}")
def get_meal_year(year: int):
    if a := meal_backend({"year": year}):
        return a
    else:
        raise HTTPException(status_code=404)
