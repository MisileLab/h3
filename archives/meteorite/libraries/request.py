from fastapi import HTTPException
from httpx import get as hget
from httpx._types import QueryParamTypes
from typing import Any

def get(url: str, params: QueryParamTypes) -> Any: # pyright: ignore[reportExplicitAny]
  r = hget(url, params=params)
  if r.is_error:
    raise HTTPException(status_code=r.status_code)
  return r.json()
