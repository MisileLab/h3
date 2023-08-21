from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from selenium import webdriver
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv

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

def timetable_backend(grader: str) -> list:
    op = Options()
    op.headless = True
    driver = webdriver.Chrome(options=op)
    driver.get("http://comci.net:4082/st")
    driver.find_element(value="sc").send_keys("선린인터넷고")
    driver.execute_script("sc2_search()")
    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.LINK_TEXT, "선린인터넷고"))
    ).click()
    driver.execute_script("sc_disp(41896)")
    WebDriverWait(driver, 10).until(
        EC.text_to_be_present_in_element((By.ID, "ba"), "1-1")
    )
    Select(driver.find_element(value="ba")).select_by_value(grader)
    bss = [x.text 
        for x in driver.find_element(value="hour")
        .find_elements(by=By.CLASS_NAME, value="내용")
    ]
    values = [[],[],[],[],[]]
    for i, i2 in enumerate(bss):
        i3 = i2.split()
        if i2 != "":
            values[i % 5].append({"name": i3[0], "teacher": i3[1]})
    driver.quit()
    return values

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
def timetable_day(grade: int, _class: int, day: int):
    if day not in [1,2,3,4,5]:
        return HTTPException(status_code=404)
    try:
        return timetable_backend(f"{grade}-{_class}")[day]
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500) from e

@app.get("/timetable/{grade}/{_class}")
def timetable(grade: int, _class: int):
    try:
        return timetable_backend(f"{grade}-{_class}")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500) from e

@app.get("/classes")
def classes():
    op = Options()
    op.headless = True
    driver = webdriver.Firefox(options=op)
    driver.get("http://comci.net:4082/st")
    driver.find_element(value="sc").send_keys("선린인터넷고")
    driver.execute_script("sc2_search()")
    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.LINK_TEXT, "선린인터넷고"))
    ).click()
    driver.execute_script("sc_disp(41896)")
    WebDriverWait(driver, 10).until(
        EC.text_to_be_present_in_element((By.ID, "ba"), "1-1")
    )
    return [x.text for x in Select(driver.find_element(value="ba")).options][1:]

@app.get("/")
def donate_plz():
    return RedirectResponse("https://github.com/sponsors/MisileLab")
