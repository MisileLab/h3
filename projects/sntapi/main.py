from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from selenium import webdriver
from pymongo.mongo_client import MongoClient
from bs4 import BeautifulSoup

from dotenv import load_dotenv
from os import environ

load_dotenv()

class Days:
    monday = 0
    tuesday = 1
    wednesday = 2
    thursday = 3
    friday = 4

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

def timetable_backend(grader: str) -> list[dict]:
    driver = webdriver.Firefox()
    driver.get("http://comci.net:4082/st")
    script = f"""
    localStorage.setItem('scmm', 선린인터넷고);
    localStorage.setItem('ba', {grader});
    """
    driver.execute_script(script)
    bs = BeautifulSoup(driver.page_source, "lxml")
    bss = bs.find_all("div > table > tbody > tr")
    print(bss)
    driver.quit()

@app.get("/meal/{year}/{month}/{day}")
def get_meal(year: int, month: int, day: int):
    if a := meal_backend({"year": year, "month": month, "day": day}):
        return a
    else:
        raise HTTPException(status_code=404)

@app.get("/meal/{year}/{month}")
def get_meal_month(year: int, month: int):
    if a := meal_backend({"year": year, "month": month}):
        return a
    else:
        raise HTTPException(status_code=404)

@app.get("/meal/{year}")
def get_meal_year(year: int):
    if a := meal_backend({"year": year}):
        return a
    else:
        raise HTTPException(status_code=404)

@app.get("/timetable/{grade}/{_class}/{day}")
def timetable_day(grade: int, _class: int, day: Days):
    pass

@app.get("/timetable/{grade}/{_class}")
def timetable(grade: int, _class: int):
    pass

@app.get("/")
def donate_plz():
    return RedirectResponse("https://github.com/sponsors/MisileLab")
