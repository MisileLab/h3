from fastapi import FastAPI, Request, status, Header, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from edgedb import create_async_client

from typing import Annotated
from dataclasses import dataclass

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
db = create_async_client()

async def check_dupe(userid: str) -> bool:
 return await db.query_single("select User {id} filter .userid = <str>$userid") != None

@app.get("/check")
@limiter.limit("10000/hour")
async def check(request: Request, userid: str) -> bool:
 return check_dupe(userid)

@app.post("/register")
@limiter.limit("10/hour")
async def register(
 request: Request,
 userid: Annotated(str | None, Header(name="id")),
 pw: Annotated(str | None, Header())
):
 if check_dupe(userid):
  return HTTPException(status_code=status.HTTP_409_CONFLICT)
 await db.query_single("create User {userid := <str>$userid, pw := <str>$pw}")

