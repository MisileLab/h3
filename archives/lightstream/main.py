from collections.abc import Coroutine
from contextlib import suppress
from typing import Any, Callable
from fastapi import FastAPI, HTTPException, Request, Response
from json import JSONDecodeError, loads

from fastapi.responses import PlainTextResponse

from libraries.pow import difficulty, generate_challenge, verify_challenge

from datetime import datetime

app = FastAPI()
proxy_ips = ["127.0.0.1"]

def verify_types(lst: list[Any]) -> bool: # pyright: ignore[reportExplicitAny]
  for i in lst: # pyright: ignore[reportAny]
    if not isinstance(i, bool):
      return False
  return True

@app.middleware("http")
async def challenge(request: Request, call_next: Callable[..., Coroutine[None, None, Response]]) -> Response:
  cl = request.client
  if cl is None:
    raise HTTPException(status_code=400, detail="Client IP not found")
  response: Response
  info = "Accept-Language={},X-Real-IP={},User-Agent={},Now={}".format(
    request.headers.get("Accept-Language"),
    request.headers.get("X-Real-IP") if cl.host in proxy_ips else cl.host,
    request.headers.get("User-Agent"),
    datetime.now().isoformat()
  )
  payload = request.headers.get("X-Payload", "")
  answer = request.headers.get("X-Answer")
  with suppress(JSONDecodeError):
    if answer is None:
      return PlainTextResponse(generate_challenge(info), 400)
    else:
      loaded = loads(answer) # pyright: ignore[reportAny]
      if (
        isinstance(loaded, list)
        and len(loaded) == difficulty # pyright: ignore[reportUnknownArgumentType]
        and verify_types(loaded) # pyright: ignore[reportUnknownArgumentType]
      ):
        v = verify_challenge(payload, loaded, info) # pyright: ignore[reportUnknownArgumentType]
        if v != "":
          response = await call_next(request)
          response.headers["X-Response"] = v
          return response
  return PlainTextResponse(generate_challenge(info), 400)

@app.get("/")
async def test():
  return "succeed :sunglasses:"
