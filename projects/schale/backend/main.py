from fastapi import FastAPI, HTTPException, status, Header, UploadFile
from edgedb import create_async_client
from tomli import loads
from pydantic import BaseModel, Field

from hashlib import sha3_512
import smtplib
from email.mime.text import MIMEText

f = FastAPI()
c = create_async_client()
config = loads("./config.toml")
key = config["security"]["key"]

def send_email(name: str, content: str, to: str):
  msg = MIMEText(
      f"{content}<br><p>from theresa, the open letter website.</p>",
      _charset='utf-8',
      _subtype='html',
  )
  msg['Subject'] = f"Confirmation email of signing {name}"
  msg['From'] = msg.get("from", "amiya@rhodes.misile.xyz")
  msg['To'] = to

  s = smtplib.SMTP(config["server"])
  try:
    s.sendmail(msg.get("from", "amiya@rhodes.misile.xyz"), [to], msg.as_string())
  except smtplib.SMTPRecipientsRefused as e:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="email invalid") from e
  s.quit()

# you need to intergrate altcha

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
  raw = await c.query_single('select theresa::Letter {tldr, signers} filter name=<str>$name limit 1', name=name)
  if raw is None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
  return openLetter(**raw)

@f.post("/theresa/sign")
async def theresa_sign(
  name: str = Header(description="name of letter"),
  email: str = Header(description="email of signer")
):
  send_email(
    name,
    f"<a href='https://misile.xyz/theresa/confirm?name={name}&email={email}&hash={sha3_512(f'{name}{email}{key}'.encode()).hexdigest()}'>click here to confirm</a>",
    email
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
  if hash != sha3_512(f"{name}{email}{key}".encode()).hexdigest():
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)
  name_signer = name_signer.split("\n")[0]
  if signature is None:
    _id = (await c.query_single('''
      insert theresa::User {
        name := <str>$name,
        email := <str>$email,
        message := <str>$message,
        hash := <str>$hash
      }
    ''', name=name_signer, email=email, message=message)).id
  else:
    _id = (await c.query_single('''
      insert theresa::User {
        name := <str>$name,
        email := <str>$email,
        message := <str>$message,
        hash := <str>$hash,
        signature := <bytes>$signature
      }
    ''', name=name_signer, email=email, message=message, signature=signature.file.read())).id
  await c.execute("update theresa::Sign filter .name = <str>$name set {signers += (select detached theresa::Letter filter .id = <std::uuid>$id)}", name=name, id=_id)
