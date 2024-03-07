import strawberry
from fastapi import FastAPI, UploadFile, status, Header, HTTPException
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter
from tomli import load
from edgedb import create_async_client

from os import mkdir
from os.path import isdir
from pathlib import Path
from secrets import SystemRandom
from typing import Optional
from datetime import datetime

c = load(open("config.toml", "rb"))
keys = c["keys"]
db = create_async_client()

if not isdir("files"):
    mkdir("files")

@strawberry.type
class User:
    name: str
    pnumber: str
    me: str
    why: str
    time: float
    portfolio: Optional[str]

@strawberry.input
class UserInput:
    name: str
    pnumber: str
    me: str
    why: str
    portfolio: Optional[str]

@strawberry.type
class Query:
    @strawberry.field
    async def infos(self, key: str) -> Optional[list[User]]:
        print("a")
        if key in keys:
            return await db.query("select User {name, pnumber, me, why, time, portfolio}")

    @strawberry.field
    async def info(self, key: str, name: str) -> Optional[User]:
        if key in keys:
            return await db.query_single("""
                                                      select User {
                                                          portfolio,
                                                          me,
                                                          name,
                                                          pnumber,
                                                          why,
                                                          time
                                                       }
                                                       filter .name = <str>$name
                                                  """, name=name)

@strawberry.type
class Mutation:
    @strawberry.field
    async def send(self, i: UserInput) -> None:
        await db.query("""
                       insert User {
                           name := <str>$name,
                           pnumber := <str>$pnumber,
                           me := <str>$me,
                           why := <str>$why,
                           time := <float64>$time,
                           portfolio := <optional str>$portfolio
                       } unless conflict on .name else (
                           update User set {
                               pnumber := <str>$pnumber,
                               me := <str>$me,
                               why := <str>$why,
                               time := <float64>$time,
                               portfolio := <optional str>$portfolio
                           }
                       )""", name=i.name, pnumber=i.pnumber, me=i.me, why=i.why, time=datetime.utcnow().timestamp(), portfolio=i.portfolio)
        return None
 
schema = strawberry.Schema(query=Query, mutation=Mutation)

graphql_app = GraphQLRouter(schema)

app = FastAPI()

origins = [
    "https://lambda.misile.xyz",
    "http://localhost",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(graphql_app, prefix="/graphql")

@app.get("/")
async def sus():
    a = SystemRandom().choice([
                   "https://www.youtube.com/watch?v=FlUKCD2G0N0",
                   "https://www.youtube.com/watch?v=jjDL_zySJv4"
               ])
    print(a)
    return RedirectResponse(a)

@app.get("/files/{key}/{name}")
async def get_file(name: str, key: str):
    if key in keys:
        return FileResponse(f"files/{name}")
    else:
        raise HTTPException(status.HTTP_403_FORBIDDEN)

@app.post("/uploadfile")
async def uploadfile(file: UploadFile) -> str:
    i = 0
    s = f"files/{SystemRandom().randint(1, 18_446_744_073_709_551_615)}_{file.filename}"
    p = Path(s)
    while p.is_file():
        if i > 100:
            print("wtf storage is alright?")
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR)
        s = f"files/{SystemRandom().randint(1, 18_446_744_073_709_551_615)}_{file.filename}"
        p = Path(s)
    p.touch()
    p.write_bytes(await file.read())
    return s.removeprefix("files/")

