import strawberry
from datetime import timezone
from fastapi import FastAPI, Request, UploadFile, status, HTTPException
from fastapi.responses import RedirectResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter
from tomli import load
from edgedb import create_async_client
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from os import mkdir
from os.path import isdir
from pathlib import Path
from secrets import SystemRandom
from typing import Optional
from datetime import datetime

c = load(open("config.toml", "rb"))
keys = c["keys"]
db = create_async_client(password=c["dbpassword"])

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
        if key in keys:
            return await db.query("select User {name, pnumber, me, why, time, portfolio}")

@strawberry.type
class Mutation:
    @strawberry.field
    async def send(self, i: UserInput) -> None:
        await db.query(
            """
                       insert User {
                           name := <str>$name,
                           pnumber := <str>$pnumber,
                           me := <str>$me,
                           why := <str>$why,
                           time := <float64>$time,
                           portfolio := <optional str>$portfolio
                       }""",
            name=i.name,
            pnumber=i.pnumber,
            me=i.me,
            why=i.why,
            time=datetime.now(timezone.utc).timestamp(),
            portfolio=i.portfolio,
        )
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

limiter = Limiter(key_func=get_remote_address, default_limits="500/hour")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.include_router(graphql_app, prefix="/graphql")

@limiter.limit("60/minute")
@app.get("/")
async def sus(request: Request):
    a = SystemRandom().choice([
                   "https://www.youtube.com/watch?v=FlUKCD2G0N0",
                   "https://www.youtube.com/watch?v=jjDL_zySJv4"
               ])
    print(a)
    return RedirectResponse(a)

@limiter.limit("300/hour")
@app.get("/files/{key}/{name}")
async def get_file(request: Request, name: str, key: str):
    if key in keys:
        return FileResponse(f"files/{name}")
    else:
        raise HTTPException(status.HTTP_403_FORBIDDEN)

@limiter.limit("30/minute")
@app.post("/uploadfile")
async def uploadfile(request: Request, file: UploadFile) -> str:
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

