from fastapi import FastAPI, HTTPException, status, Header, UploadFile
from edgedb import create_async_client
from tomli import loads
from pydantic import BaseModel, Field
from boto3 import client

from hashlib import sha3_256
from pathlib import Path
from dataclasses import asdict
from base64 import b64encode

f = FastAPI()
c = create_async_client()
config = loads(Path("./config.toml").read_text())
smtp_client = client('ses', region_name="eu-west-2", aws_access_key_id=config["aws"]["key"], aws_secret_access_key=config["aws"]["secret_key"])
key = config["security"]["key"]

# manual pow (integrate with binah)

def send_email(name: str, content: str, to: list[str]):
  print(f"send email to {to}")
  resp = smtp_client.send_email(
    Destination={"ToAddresses": to},
    Message={
      'Body': {'Html': {'Charset': 'UTF-8', 'Data': content + "<p>From theresa, open letter platform</p>"}},
      'Subject': {'Charset': 'UTF-8', 'Data': f'Sign the open letter, {name}'}
    },
    Source="theresa <theresa@rhodes.misile.xyz>"
  )
  print(resp)

class Signer(BaseModel):
  name: str = Field(description="name of signer")
  email: str = Field(description="email of signer")
  message: str = Field(description="message of signer", default = "")
  signature: bytes | None = Field(description="signature of signer", default=None)

class openLetter(BaseModel):
  name: str = Field(description="name of letter")
  tldr: str = Field(description="one line of letter")
  signers: list[Signer] = Field(description="list of signers", default=[])

@f.get("/theresa/info")
async def theresa_info(
  name: str = Header(description="name of letter")
) -> openLetter:
  raw = asdict(await c.query_single('select theresa::Letter {name, tldr, signers} filter .name=<str>$name limit 1', name=name))
  if raw is None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
  def process_data(x: dict):
    x["signature"] = b64encode(x["signature"])
    return x
  raw["signers"] = [process_data(asdict(await c.query_single('select theresa::User {name, message, signature, email} filter .id=<uuid>$id', id=i['id']))) for i in raw["signers"]]
  return openLetter(**raw)

@f.post("/theresa/sign")
async def theresa_sign(
  name: str = Header(description="name of letter"),
  email: str = Header(description="email of signer")
):
  print(f'{name}{email}{key}')
  send_email(
    name,
    f"<a href='https://misile.xyz/theresa/confirm?name={name}&email={email}&hash={sha3_256(f'{name}{email}{key}'.encode()).hexdigest()}'>click here to confirm</a>",
    [email]
  )

@f.post("/theresa/confirm")
async def theresa_confirm(
  name: str = Header(description="name of letter"),
  name_signer: str = Header(description="name of signer"),
  email: str = Header(description="email of signer"),
  hash: str = Header(description="hash of signer"),
  message: str = Header(description="message of signer"),
  signature: UploadFile | None = None
):
  if hash != sha3_256(f"{name_signer}{email}{key}".encode()).hexdigest():
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
  name_signer = name_signer.split("\n")[0]
  dupe_id = await c.query_single('select theresa::User {id} filter .name=<str>$name limit 1', name=name_signer)
  if dupe_id is None:
    if signature is None:
      _id = (await c.query_single('''
        insert theresa::User {
          name := <str>$name,
          email := <str>$email,
          message := <str>$message,
          hash := <str>$hash
        }
      ''', name=name_signer, email=email, message=message, hash=hash)).id
    else:
      _id = (await c.query_single('''
        insert theresa::User {
          name := <str>$name,
          email := <str>$email,
          message := <str>$message,
          hash := <str>$hash,
          signature := <bytes>$signature
        }
      ''', name=name_signer, email=email, message=message, signature=signature.file.read(), hash=hash)).id
  elif signature is None:
    _id = (await c.query_single('''update theresa::User filter .id = <uuid>$id set {
      name := <str>$name,
      email := <str>$email,
      message := <str>$message,
      hash := <str>$hash
    }''', name=name_signer, email=email, message=message, hash=hash, id=dupe_id.id)).id
  else:
    _id = (await c.query_single('''update theresa::User filter .id = <uuid>$id set {
      name := <str>$name,
      email := <str>$email,
      message := <str>$message,
      hash := <str>$hash,
      signature := <bytes>$signature
    }''', name=name_signer, email=email, message=message, hash=hash, signature=signature.file.read(), id=dupe_id.id)).id
  await c.execute("update theresa::Letter filter .name = <str>$name set {signers += (select detached theresa::User filter .id = <std::uuid>$id)}", name=name, id=_id)
