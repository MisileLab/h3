from libraries.initializer import initializer
from libraries.email import send_email

from fastapi import HTTPException, status, Header, UploadFile, APIRouter, Form
from pydantic import BaseModel, Field
from httpx import AsyncClient

from hashlib import sha3_256
from dataclasses import asdict

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

@router.get("/info")
async def info(
  name: str = Header(description="name of letter")
) -> openLetterPublic:
  raw = asdict(
    await initializer.c.query_single('select theresa::Letter {name, tldr, file, signers: {id}} filter .name=<str>$name limit 1', name=name)
  )
  if raw is None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
  raw["signer"] = len(raw["signers"])
  del raw["signers"]
  return openLetterPublic(**raw)

@router.get("/info/signers")
async def info_signers(
  name: str = Header(description="name of letter")
) -> list[SignerBase]:
  raw = asdict(
    await initializer.c.query_single('select theresa::Letter {signers: {name, signature}} filter .name=<str>$name limit 1', name=name)
  )
  if raw is None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
  signers = raw["signers"]
  return [SignerBase(**i) for i in signers]

@router.get("/info/signer")
async def info_signer(
  name: str = Header(description="name of letter"),
  name_signer: str = Header(description="name of signer")
) -> Signer:
  raw = asdict(
    await initializer.c.query_single(
      'select theresa::Letter {signers filter .name=<str>$name_signer limit 1} filter .name=<str>$name limit 1',
      name=name,
      name_signer=name_signer
    )
  )
  if raw is None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
  return Signer(**raw["signers"][0])

@router.post("/sign")
async def sign(
  name: str = Form(description="name of letter"),
  email: str = Form(description="email of signer"),
  hcaptcha_response: str = Form(description="h-captcha response")
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
  raw = asdict(
    await initializer.c.query_single('select theresa::Letter {id} filter .name=<str>$name limit 1', name=name)
  )
  if raw is None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
  send_email(
    name,
    f"<a href='https://misile.xyz/theresa/confirm?name={name}&email={email}&hash={sha3_256(f'{name}{email}{initializer.key}'.encode()).hexdigest()}'>click here to confirm</a>",
    [email]
  )

@router.post("/confirm")
async def confirm(
  name: str = Header(description="name of letter"),
  name_signer: str = Header(description="name of signer"),
  email: str = Header(description="email of signer"),
  hash: str = Header(description="hash of signer"),
  message: str = Header(description="message of signer"),
  signature: UploadFile | None = None
):
  if hash != sha3_256(f"{name_signer}{email}{initializer.key}".encode()).hexdigest():
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
  name_signer = name_signer.split("\n")[0]
  dupe_id = await initializer.c.query_single('select theresa::User {id} filter .name=<str>$name limit 1', name=name_signer)
  if dupe_id is None:
    if signature is None:
      _id = (await initializer.c.query_single('''
        insert theresa::User {
          name := <str>$name,
          email := <str>$email,
          message := <str>$message,
          hash := <str>$hash
        }
      ''', name=name_signer, email=email, message=message, hash=hash)).id
    else:
      _id = (await initializer.c.query_single('''
        insert theresa::User {
          name := <str>$name,
          email := <str>$email,
          message := <str>$message,
          hash := <str>$hash,
          signature := <bytes>$signature
        }
      ''', name=name_signer, email=email, message=message, signature=signature.file.read(), hash=hash)).id
  elif signature is None:
    _id = (await initializer.c.query_single('''update theresa::User filter .id = <uuid>$id set {
      name := <str>$name,
      email := <str>$email,
      message := <str>$message,
      hash := <str>$hash
    }''', name=name_signer, email=email, message=message, hash=hash, id=dupe_id.id)).id
  else:
    _id = (await initializer.c.query_single('''update theresa::User filter .id = <uuid>$id set {
      name := <str>$name,
      email := <str>$email,
      message := <str>$message,
      hash := <str>$hash,
      signature := <bytes>$signature
    }''', name=name_signer, email=email, message=message, hash=hash, signature=signature.file.read(), id=dupe_id.id)).id
  await initializer.c.execute(
    "update theresa::Letter filter .name = <str>$name set {signers += (select detached theresa::User filter .id = <std::uuid>$id)}",
    name=name,
    id=_id
  )
