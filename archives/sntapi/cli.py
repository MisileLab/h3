from pymongo.mongo_client import MongoClient
from dotenv import load_dotenv
from os import environ

load_dotenv()

db = MongoClient(host=environ["HOST"], port=int(environ["PORT"]), username=environ["USERNAME"], password=environ["PASSWORD"]) # noqa: E501

d = db.get_database("local")
d2 = d.get_collection("sntmeal")
b = []

while True:
    a = input()
    if a == "push":
        d2.insert_many(b)
        break
    try:
        c, d, e = map(int, a.split(" "))
    except ValueError:
        print("try")
    f = []
    print("====data====")
    while True:
        try:
            g = input()
        except UnicodeDecodeError:
            print("try again")
            continue
        if g == "end":
            print("end it")
            break
        f.append(g)
    b.append({"year": c, "month": d, "day": e, "meal": f})
