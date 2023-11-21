from typing import List, Optional
import strawberry
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strawberry.asgi import GraphQL
from tomli import loads
from pymongo import MongoClient
from requests import get
from pathlib import Path
import hashlib
import colorsys

config = loads(Path("config.toml").read_text())
db = MongoClient(config["MONGO_URI"])["schoolfinder"]


@strawberry.type
class UserType:
    student = "student"
    teacher = "teacher"


@strawberry.type
class Answer:
    name: str
    description: str
    confirmed: bool


@strawberry.type
class Question:
    name: str
    description: str
    answers: List[Answer]


@strawberry.type
class Review:
    name: str
    description: str
    stars: float


@strawberry.type
class Account:
    name: str
    email: str
    password: str
    user_type: str
    question: int
    answer: int
    solved: int
    reviews: List[Review]
    school: str
    description: str
    hex_codes: List[str]


@strawberry.type
class School:
    name: str
    description: str
    callnumber: str
    address: str
    homepage: str
    questions: List[Question]
    school_type: str


def hsl_to_rgb(h, s, l):
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255),
                                        int(b * 255))


def generate_gradient(uid, type='diagonal'):
    if uid:
        n = int(hashlib.sha256(uid.encode()).hexdigest(), 16) % 360
        h = n / 360.0
        s = 0.95
        l = 0.5

        c1_hex = hsl_to_rgb(h, s, l)

        triad_h = (h + 120) % 360
        c2_hex = hsl_to_rgb(triad_h / 360.0, s, l)

        return c1_hex, c2_hex

    raise TypeError('uid is required')


@strawberry.type
class Query:

    @strawberry.field
    def user(self, email: str) -> Optional[Account]:
        data = db["login"].find_one({"email": email})
        if data is None:
            return None
        return Account(name=data["name"],
                       email=data["email"],
                       password="",
                       user_type=data["user_type"],
                       question=data["question"],
                       answer=data["answer"],
                       solved=data["solved"],
                       reviews=[
                           Review(stars=i["stars"],
                                  name=i["name"],
                                  description=i["description"])
                           for i in data["reviews"]
                       ],
                       school=data["school"],
                       description=data["description"],
                       hex_codes=data["hex_codes"])

    @strawberry.field
    def school(self, name: str) -> List[School]:
        a = get("https://open.neis.go.kr/hub/schoolInfo",
                params={
                    "KEY": config["NEIS_KEY"],
                    "TYPE": "json",
                    "SCHUL_NM": name,
                    "SCHUL_KND_SC_NM": "고등학교"
                })
        a.raise_for_status()
        first_stage = [
            School(name=i["SCHUL_NM"],
                   description="",
                   callnumber=i["ORG_TELNO"],
                   address=i["ORG_RDNMA"],
                   homepage=i["HMPG_ADRES"],
                   questions=[],
                   school_type=i["HS_SC_NM"])
            for i in a.json()["schoolInfo"][1]["row"]
        ]
        for i in first_stage:
            data = db["info"].find_one({"name": name})
            if data is None:
                data = db["info"].insert_one({
                    "name": name,
                    "description": "",
                    "callnumber": i.callnumber,
                    "address": i.address,
                    "homepage": i.homepage,
                    "questions": [],
                    "comments": [],
                    "school_type": i.school_type
                })
                continue
            i.description = data["description"]
            i.questions = [
                Question(name=i["name"],
                         description=i["description"],
                         answers=[
                             Answer(name=i2["name"],
                                    description=i2["description"],
                                    confirmed=i2["confirmed"])
                             for i2 in i["answers"]
                         ]) for i in data["questions"]
            ]
        return first_stage


@strawberry.type
class Mutation:

    @strawberry.mutation
    def register(self, name: str, email: str, password: str,
                 user_type: str) -> Account:
        if db["login"].find_one({"email": email}) is not None:
            raise ValueError("Already registered")
        db["login"].insert_one({
            "name": name,
            "email": email,
            "password": password,
            "user_type": user_type,
            "question": 0,
            "answer": 0,
            "solved": 0,
            "reviews": [],
            "school": ""
        })
        return Account(name=name,
                       email=email,
                       password="",
                       user_type=user_type,
                       question=0,
                       answer=0,
                       solved=0,
                       reviews=[],
                       school="",
                       description="",
                       hex_codes=list(generate_gradient(email)))

    @strawberry.mutation
    def login(self, email: str, password: str) -> Optional[Account]:
        data = db["login"].find_one({"email": email, "password": password})
        if data is None:
            return None
        return Account(name=data["name"],
                       email=data["email"],
                       password="",
                       user_type=data["user_type"],
                       question=data["question"],
                       answer=data["answer"],
                       solved=data["solved"],
                       reviews=data["reviews"],
                       school=data["school"],
                       description=data["description"],
                       hex_codes=data["hex_codes"])

    @strawberry.mutation
    def set_description(self, email: str, password: str,
                        description: str) -> None:
        data = db["login"].find_one({"email": email, "password": password})
        if data is None:
            raise ValueError("Invalid email or password")
        db["login"].update_one({
            "email": email,
            "password": password
        }, {"$set": {
            "description": description
        }})

    @strawberry.mutation
    def add_review(self, target_email: str, email: str, password: str,
                   description: str, stars: float) -> Review:
        user = db["login"].find_one({"email": email, "password": password})
        target_user = db["login"].find_one({"email": target_email})
        if user is None:
            raise ValueError("Invalid email or password")
        if target_user is None:
            raise ValueError("Invalid target email")
        data = {
            "name": user["name"],
            "description": description,
            "stars": stars
        }
        if data in target_user["reviews"]:
            target_user["reviews"].remove(data)
        target_user["reviews"].append(data)
        db["login"].update_one({"email": target_email},
                               {"$set": {
                                   "reviews": target_user["reviews"]
                               }})
        return Review(name=user["name"], description=description, stars=stars)

    @strawberry.mutation
    def add_question(self, target_name: str, email: str, password: str,
                     description: str) -> Question:
        user = db["login"].find_one({"email": email, "password": password})
        school = db["info"].find_one({"name": target_name})
        if user is None:
            raise ValueError("Invalid email or password")
        if school is None:
            raise ValueError(
                "Invalid school name (maybe you need to search school once)")
        data = {
            "name": user["name"],
            "description": description,
            "answers": []
        }
        school["questions"].append(data)
        db["info"].update_one({"name": target_name},
                              {"$set": {
                                  "questions": school["questions"]
                              }})
        return Question(name=user["name"], description=description, answers=[])

    @strawberry.mutation
    def add_answer(self, target_school: str, target_id: int, email: str,
                   password: str, description: str) -> Answer:
        user = db["login"].find_one({"email": email, "password": password})
        school = db["info"].find_one({"name": target_school})
        if user is None:
            raise ValueError("Invalid email or password")
        if school is None:
            raise ValueError(
                "Invalid school name (maybe you need to search school once)")
        data = {
            "name": user["name"],
            "description": description,
            "confirmed": False
        }
        school["questions"][target_id]["answers"].append(data)
        db["info"].update_one({"name": target_school},
                              {"$set": {
                                  "questions": school["questions"]
                              }})
        return Answer(name=user["name"],
                      description=description,
                      confirmed=False)

    @strawberry.mutation
    def confirm(self, email: str, password: str, target_school: str,
                target_id: int, confirm: bool) -> Answer:
        user = db["login"].find_one({"email": email, "password": password})
        school = db["info"].find_one({"name": target_school})
        if user is None:
            raise ValueError("Invalid email or password")
        if school is None:
            raise ValueError(
                "Invalid school name (maybe you need to search school once)")
        if user["school"] != target_school:
            raise ValueError("You are not allowed to confirm")
        school["questions"][target_id]["answers"][0]["confirmed"] = confirm
        db["info"].update_one({"name": target_school},
                              {"$set": {
                                  "questions": school["questions"]
                              }})
        return Answer(
            name=school["questions"][target_id]["answers"][0]["name"],
            description=school["questions"][target_id]["answers"][0]
            ["description"],
            confirmed=confirm)


schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_app = GraphQL(schema=schema)
app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=False,
                   allow_methods=["*"],
                   allow_headers=["*"])
app.add_route("/graphql", graphql_app)
app.add_websocket_route("/graphql", graphql_app)
