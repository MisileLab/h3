from httpx import post
from edgedb import create_async_client
from fastapi import FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.mount("/files", StaticFiles(directory="files"), name="files")

db = create_async_client()
app = FastAPI()

@app.get("/")
@limiter.limit("1/second")
async def get_gpg(jwtv: str = Header(default="")):
 r = post("https://schale.misile.xyz/verify", headers={'jwtv':jwtv})
 if not r.is_success:
  raise HTTPException(status_code=r.status_code)
 return await db.query_json("select User {groups} filter .userid = <str>$userid", userid=r.text)

