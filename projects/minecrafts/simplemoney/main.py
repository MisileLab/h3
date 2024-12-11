from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import FastAPI, Header, status, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from edgedb import Object, create_async_client
from loguru import logger

from typing import Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path
from json.decoder import JSONDecodeError
from collections import defaultdict

db = create_async_client()
app = FastAPI()
wss: defaultdict[str, list[WebSocket]] = defaultdict(list)
whitelisted_ip = Path('./whitelisted_ip').read_text().split('\n') if Path('./whitelisted_ip').is_file() else ['127.0.0.1']

@app.middleware('http')
async def validate_ip(request: Request, call_next):
  # Get client IP
  if request.client is None:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"message": "something wrong"})
  ip = request.headers.get('X-Forwarded-For', '').split(',')[0] or request.client.host
  logger.debug(ip)
  
  # Check if IP is allowed
  if ip not in whitelisted_ip:
    data = {
      'message': f'IP {ip} is not allowed to access this resource.'
    }
    return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content=data)

  # Proceed if IP is allowed
  return await call_next(request)

@dataclass
class Account:
  money: int
  name: str
  password: Optional[str]
  transactions: list[str] = field(default_factory=list)

@dataclass
class Transaction:
  amount: int
  to: str
  received: str

def conv_to_dict(v: Object | None, status_code: int = 400) -> dict:
  if v is None:
    raise HTTPException(status_code=status_code, detail="result of query is None (probably name or password is invalid)")
  return asdict(v)

def verify_pw(hash: str, v: str) -> bool:
  try:
    ph = PasswordHasher()
    ph.verify(hash, v)
  except VerifyMismatchError:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="name or password is mismatched")
  return ph.check_needs_rehash(hash)

@app.get("/")
async def ping():
  return {"ping": "pong"}

@app.get("/user")
async def get_user(
  name: str = Header()
) -> Account:
  _user = conv_to_dict(await db.query_single("select Account {name, money, transactions: {id}} filter .name = <str>$name limit 1", name=name))
  _user['transactions'] = [str(i['id']) for i in _user['transactions']]
  return Account(password=None, **_user)

@app.get("/transaction")
async def get_transaction(
  id: str = Header()
) -> Transaction:
  return Transaction(**conv_to_dict(await db.query_single("select Transaction {amount, to, received} filter .id = <std::uuid>$id limit 1", id=id)))

@app.post("/account/send")
async def send_money(
  name: str = Header(),
  password: str = Header(),
  to: str = Header(),
  amount: int = Header()
) -> PlainTextResponse:
  if name == to:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="can't send to same account")
  _account: dict = conv_to_dict(await db.query_single("select Account {name, money, password} filter .name = <str>$name limit 1", name=name))
  if verify_pw(_account['password'], password):
    await db.query_single("update Account filter .name = <str>$name set {password := <str>$password}", password=PasswordHasher().hash(password))
  _account_to: dict = conv_to_dict(await db.query_single("select Account {name, money} filter .name = <str>$name limit 1", name=to))
  account = Account(**_account, password=None)
  account_to = Account(**_account_to, password=None)
  if account.money - amount < 0 or amount <= 0:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="money amount is less than 0 or your money is not sufficient")
  t = (await db.query_single("""insert Transaction {
    to := <str>$to,
    received := <str>$received,
    amount := <int64>$amount
  }""", to=account.name, received=account_to.name, amount=amount)).id
  await db.query_single("""update Account filter .name = <str>$name set {
    money := <int64>$amount,
    transactions += (select detached Transaction filter .id = <std::uuid>$id)
  }""", name=account.name, id=t, amount=account.money-amount)
  await db.query_single("""update Account filter .name = <str>$name set {
    money := <int64>$amount,
    transactions += (select detached Transaction filter .id = <std::uuid>$id)
  }""", name=account_to.name, id=t, amount=account_to.money+amount)
  _t = str(t)
  for i in wss.get(to, []):
    await i.send_json({"type": "receive", "id": _t})
  for i in wss.get(name, []):
    await i.send_json({"type": "send", "id": _t})
  return PlainTextResponse(_t)

@app.websocket("/event")
async def event(ws: WebSocket):
  await ws.accept()
  try:
    data = await ws.receive_json()
  except JSONDecodeError:
    await ws.send_json({"status_code": 400, "detail": "Data is invalid"})
    await ws.close()
    return
  name = data['name']
  try:
    if verify_pw(
      conv_to_dict(await db.query_single("select Account {name, money} filter .name = <str>$name", name=name))['password'],
      data['password']
    ):
      await db.query_single("update Account filter .name = <str>$name {password := <str>$password}", password=PasswordHasher().hash(data['password']))
  except HTTPException:
    await ws.send_json({"status_code": 400, "detail": "Login failed"})
    await ws.close()
    return
  wss[name].append(ws)
  try:
    while True:
      data = await ws.receive_text()
      await ws.send_json({'ping': 'pong'})
  except WebSocketDisconnect:
    wss[name].remove(ws)
    if len(wss[name]) == 0:
      del wss[name]
