from libraries.initializer import initializer
from libraries.email import send_email
from libraries.lib import nullVerify

from fastapi import HTTPException, status, Header, APIRouter, Form
from satellite_py import generate_error_responses
from pydantic import BaseModel, Field
from httpx import AsyncClient
from blake3 import blake3

from dataclasses import asdict
from typing import Annotated
from uuid import UUID

router = APIRouter()

class SignerBase(BaseModel):
  name: str = Field(description="name of signer")
  signature: str | None = Field(description="signature of signer", default=None)

class Signer(SignerBase):
  email: str = Field(description="email of signer")
  message: str = Field(description="message of signer", default = "")

class openLetterBase(BaseModel):
  name: str = Field(description="name of letter")
  tldr: str = Field(description="one line of letter")
  file: str = Field(description="link of file")

class openLetter(openLetterBase):
  signers: list[Signer] = Field(description="list of signers", default=[])

class openLetterPublic(openLetterBase):
  signer: int = Field(description="count of signers", default=0)

NOT_FOUND_RESP= generate_error_responses([404])

@router.get("/info", responses=NOT_FOUND_RESP)
async def info(
  name: Annotated[str, Header(description="name of letter")]
) -> openLetterPublic:
  raw = await initializer.c.query_single('select theresa::Letter {name, tldr, file, signers: {id}} filter .name=<str>$name limit 1', name=name)
  if raw is None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
  raw = asdict(raw)
  raw["signer"] = len(raw["signers"])
  del raw["signers"]
  return openLetterPublic(**raw)

@router.get("/info/signers", responses=NOT_FOUND_RESP)
async def info_signers(
  name: Annotated[str, Header(description="name of letter")]
) -> list[SignerBase]:
  raw = await initializer.c.query_single('select theresa::Letter {signers: {name, signature}} filter .name=<str>$name limit 1', name=name)
  if raw is None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
  raw = asdict(raw)
  signers = raw["signers"]
  return [SignerBase(**i) for i in signers]

@router.get("/info/signer", responses=NOT_FOUND_RESP)
async def info_signer(
  name: Annotated[str, Header(description="name of letter")],
  name_signer: Annotated[str, Header(description="name of signer")]
) -> Signer:
  raw = await initializer.c.query_single(
    'select theresa::Letter {signers: {name, signature, email, message} filter .name=<str>$name_signer limit 1} filter .name=<str>$name limit 1',
    name=name,
    name_signer=name_signer
  )
  if raw is None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
  raw = asdict(raw)
  return Signer(**raw["signers"][0])

@router.post("/sign", responses=generate_error_responses([403, 404]))
async def sign(
  name: Annotated[str, Form(description="name of letter")],
  email: Annotated[str, Form(description="email of signer")],
  hcaptcha_response: Annotated[str, Form(description="h-captcha response", convert_underline=False, alias="h-captcha-response")]
):
  async with AsyncClient() as client:
    if not (await client.post(
      "https://hcaptcha.com/siteverify",
      data={
        "secret": initializer.config["security"]["h_captcha"],
        "response": hcaptcha_response
      }
    )).json()["success"]:
      raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
  raw = await initializer.c.query_single('select theresa::Letter {id} filter .name=<str>$name limit 1', name=name)
  if raw is None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
  raw = asdict(raw)
  params = f"?name={name}&email={email}&hash={blake3(f'{name}{email}{initializer.key}'.encode()).hexdigest()}"
  content = f"<a href='https://misile.xyz/theresa/confirm{params}'>click here to confirm (normal)</a>\n"
  content += f"<a herf='http://b723cfcf6psmade7vqldtbc332nhhrwy52wvka3afy5s5257pzbqswid.onion/theresa/confirm{params}'>click here to confirm (tor)</a>"
  send_email(
    name,
    content,
    [email]
  )

@router.post("/confirm", responses=generate_error_responses([403]))
async def confirm(
  name: Annotated[str, Header(description="name of letter")],
  name_signer: Annotated[str, Header(description="name of signer")],
  email: Annotated[str, Header(description="email of signer")],
  hash: Annotated[str, Header(description="hash of signer")],
  message: Annotated[str, Header(description="message of signer")],
  signature: Annotated[str | None, Header(description="signature of signer")] = None
):
  if hash != blake3(f"{name}{email}{initializer.key}".encode()).hexdigest():
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
  name_signer = name_signer.split("\n")[0]
  dupe_id = await initializer.c.query_single('select theresa::User {id} filter .name=<str>$name limit 1', name=name_signer)
  _id: UUID
  if dupe_id is None:
    if signature is None:
      _id = nullVerify(await initializer.c.query_single( # pyright: ignore[reportUnknownVariableType]
      '''
        insert theresa::User {
          name := <str>$name,
          email := <str>$email,
          message := <str>$message,
          hash := <str>$hash
        }
      ''', name=name_signer, email=email, message=message, hash=hash)).id # pyright: ignore[reportAttributeAccessIssue]
    else:
      _id = nullVerify(await initializer.c.query_single( # pyright: ignore[reportUnknownVariableType]
      '''
        insert theresa::User {
          name := <str>$name,
          email := <str>$email,
          message := <str>$message,
          hash := <str>$hash,
          signature := <str>$signature
        }
      ''', name=name_signer, email=email, message=message, signature=signature, hash=hash)).id # pyright: ignore[reportAttributeAccessIssue]
  elif signature is None:
    _id = nullVerify(await initializer.c.query_single( # pyright: ignore[reportUnknownVariableType]
    '''update theresa::User filter .id = <uuid>$id set {
      name := <str>$name,
      email := <str>$email,
      message := <str>$message,
      hash := <str>$hash
    }''', name=name_signer, email=email, message=message, hash=hash, id=dupe_id.id)).id # pyright: ignore[reportAttributeAccessIssue, reportUnknownArgumentType]
  else:
    _id = nullVerify(await initializer.c.query_single( # pyright: ignore[reportUnknownVariableType]
    '''update theresa::User filter .id = <uuid>$id set {
      name := <str>$name,
      email := <str>$email,
      message := <str>$message,
      hash := <str>$hash,
      signature := <str>$signature
    }''', name=name_signer, email=email, message=message, hash=hash, signature=signature, id=dupe_id.id)).id # pyright: ignore[reportAttributeAccessIssue, reportUnknownArgumentType]
  _ = await initializer.c.query(
    "update theresa::Letter filter .name = <str>$name set {signers += (select detached theresa::User filter .id = <std::uuid>$id)}",
    name=name,
    id=_id
  )
