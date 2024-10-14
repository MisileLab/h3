from fastapi import FastAPI, Header
from edgedb import create_async_client
from tomli import loads
from pydantic import BaseModel, Field

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
  name: str = Field(description="name of letter")
) -> openLetter | None:
  await c.query_single('')

