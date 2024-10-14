from fastapi import FastAPI, HTTPException, status, Header
from edgedb import create_async_client
from tomli import loads
from pydantic import BaseModel, Field

from hashlib import sha3_512

# you need to intergrate altcha, I need to sleep after some minutes so I just poc it

f = FastAPI()
c = create_async_client()
key = loads("./config.toml")["security"]["key"]

class Signer(BaseModel):
  name: str = Field(description="name of signer")
  email: str = Field(description="email of signer")
  message: str = Field(description="message of signer", default = "")

class openLetter(BaseModel):
  name: str = Field(description="name of letter")
  tldr: str = Field(description="one line of letter")
  signers: list[Signer] = Field(description="list of signers", default=[])

@f.get("/theresa/info")
async def theresa_info(
  name: str = Header(description="name of letter")
) -> openLetter:
  raw = await c.query_single('select Letter {tldr, signers} filter name=<str>$name limit 1', name=name)
  if raw is None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
  return openLetter(**raw)

@f.post("/theresa/sign")
async def theresa_sign(
  name: str = Header(description="name of letter"),
  email: str = Header(description="email of signer")
) -> str:
  # implement auth email logic
  return sha3_512(f"{name}{email}{key}".encode()).hexdigest()

@f.post("/theresa/confirm")
async def theresa_confirm(
  name: str = Header(description="name of letter"),
  name_signer: str = Header(description="name of signer"),
  email: str = Header(description="email of signer"),
  hash: str = Header(description="hash of signer"),
  message: str = Header(description="message of signer")
):
  if hash != sha3_512(f"{name}{email}{key}".encode()).hexdigest():
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
  # verify email and actually makes sign
