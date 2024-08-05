from fastapi import FastAPI, Request, status, Header, HTTPException
from fastapi import Path as FPath
from fastapi.middleware.cors import CORSMiddleware
from edgedb import create_async_client
from jwt import decode, encode, exceptions

from typing import Annotated
from pathlib import Path
from datetime import timedelta, timezone, datetime
from hashlib import sha3_256

KEY = Path("./KEY").read_text()
TIMEOUT = timedelta(weeks=4) # four weeks
ALG = "HS256"
ORIGINS = ["null", "*"]

app = FastAPI()
app.add_middleware(
  CORSMiddleware,
  allow_origins=ORIGINS,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"]
)
db = create_async_client()

def utcnow():
  return datetime.now(tz=timezone.utc)

async def check_dupe(userid: str) -> bool:
  return await db.query("select User {id} filter .userid = <str>$userid", userid=userid) != []

@app.get("/check/{userid}")
async def check(userid: str = FPath(description="user's id")) -> bool:
  """check user's id is duplicated or not"""
  return await check_dupe(userid)

@app.post("/login")
async def login(
  userid: Annotated[str | None, Header(alias="id", description="user's id")] = None,
  pw: Annotated[str | None, Header(description="user's password")] = None
) -> str:
  """
  login with user's id and password.
  if header is invalid, return 400
  if login failed, return 401
  """
  if userid is None or pw is None:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
  if await db.query("select User {id} filter .userid = <str>$userid and .pw = <str>$pw", userid=userid,pw=(sha3_256(pw.encode('utf8'))).hexdigest()) == []:
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
  return encode({"exp": utcnow() + TIMEOUT, "id": userid}, KEY, algorithm=ALG)

@app.post("/verify")
async def verify(
  jwtv: Annotated[str | None, Header(alias="jwt", description="auth jwt value")] = None
):
  """
  verify jwt is valid
  if header is invalid or jwt failed to decode, return 400
  if signature expired, return 403
  """
  if jwtv is None:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
  try:
    j = decode(jwtv, KEY, algorithms=[ALG])
    return j["id"]
  except exceptions.ExpiredSignatureError:
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
  except exceptions.DecodeError:
    print('decode')
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

@app.post("/register")
async def register(
  userid: Annotated[str | None, Header(alias="id", description="user's id")] = None,
  pw: Annotated[str | None, Header(description="user's password")] = None
):
  """
  register with id and password
  if header is invalid, return 400
  if same id exists, return 409
  """
  if userid is None or pw is None:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
  if await check_dupe(userid):
    return HTTPException(status_code=status.HTTP_409_CONFLICT)
  await db.query_single("insert User {userid := <str>$userid, pw := <str>$pw}",userid=userid,pw=sha3_256(pw.encode('utf8')).hexdigest())

