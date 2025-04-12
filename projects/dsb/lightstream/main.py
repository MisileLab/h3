from fastapi import FastAPI, HTTPException, Header, Request

from libraries.pow import generate_challenge, verify_challenge

from typing import Annotated
from datetime import datetime

app = FastAPI()
proxy_ips = ["127.0.0.1"]

@app.get("/hash")
async def challenge(request: Request):
  cl = request.client
  if cl is None:
    raise HTTPException(status_code=400, detail="Client IP not found")
  return generate_challenge(
    "Accept-Language={},X-Real-IP={},User-Agent={},Now={}".format(
      request.headers.get("Accept-Language"),
      request.headers.get("X-Real-IP") if cl.host in proxy_ips else cl.host,
      request.headers.get("User-Agent"),
      datetime.now().isoformat()
    )
  )

