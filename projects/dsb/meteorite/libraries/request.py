from fastapi import HTTPException
from httpx import get as hget
from httpx._types import QueryParamTypes

def get(url: str, params: QueryParamTypes) -> dict:
  r = hget(url, params=params)
  if r.is_error:
    raise HTTPException(status_code=r.status_code)
  return r.json()
