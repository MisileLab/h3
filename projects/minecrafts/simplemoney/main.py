from fastapi import FastAPI, Header, status, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from edgedb import Object, create_async_client
from loguru import logger

from typing import Optional
from dataclasses import dataclass, asdict, field
from hashlib import sha3_512
from pathlib import Path
from secrets import token_bytes
from json.decoder import JSONDecodeError

db = create_async_client()
app = FastAPI()
if not Path('./salt').is_file():
  Path('./salt').write_bytes(token_bytes(128))
salt = Path('./salt').read_bytes()
wss: dict[str, list[WebSocket]] = {}
whitelisted_ip = Path('./whitelisted_ip').read_text().split('\n') if Path('./whitelisted_ip').is_file() else ['127.0.0.1']

@app.middleware('http')
async def validate_ip(request: Request, call_next):
  # Get client IP
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
  transactions: list[str] = field(default_factory=lambda: [])

@dataclass
class Transaction:
  amount: int
  to: str
  received: str

def conv_to_dict(v: Object | None, status_code: int = 400) -> dict:
  if v is None:
    raise HTTPException(status_code=status_code, detail="result of query is None (probably name or password is invalid)")
  return asdict(v)

@app.get("/")
async def ping():
  return {"ping": "pong"}

@app.get("/user")
async def get_user(
  name: str | None = Header()
) -> Account:
  if not isinstance(name, str):
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
  _user = conv_to_dict(await db.query_single("select Account {name, money, transactions: {id}} filter .name = <str>$name limit 1", name=name))
  _user['transactions'] = [str(i['id']) for i in _user['transactions']]
  del _user['id']
  user: Account = Account(password=None, **_user)
  return user

@app.get("/transaction")
async def get_transaction(
  id: str | None = Header()
) -> Transaction:
  if not isinstance(id, str):
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
  transaction: dict = conv_to_dict(await db.query_single("select Transaction {amount, to, received} filter .id = <std::uuid>$id limit 1", id=id))
  del transaction['id']
  return Transaction(**transaction)

@app.post("/account/send")
async def send_money(
  name: str | None = Header(),
  password: str | None = Header(),
  to: str | None = Header(),
  amount: int | None = Header()
) -> PlainTextResponse:
  if not (isinstance(name, str) and isinstance(password, str) and isinstance(amount, int) and isinstance(to, str)):
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
  if name == to:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="can't send to same account")
  sha_hash = sha3_512(password.encode('utf8') + salt).hexdigest()
  _account: dict = conv_to_dict(await db.query_single("select Account {name, money} filter .name = <str>$name and .password = <str>$password limit 1", name=name, password=sha_hash))
  _account_to: dict = conv_to_dict(await db.query_single("select Account {name, money} filter .name = <str>$name limit 1", name=to))
  del _account['id']
  del _account_to['id']
  account = Account(**_account, password=None)
  account_to = Account(**_account_to, password=None)
  if account.money - amount < 0 or amount <= 0:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="money amount is less than 0 or your money is not sufficient")
  t = (await db.query_single('insert Transaction {to := <str>$to, received := <str>$received, amount := <int64>$amount}', to=account.name, received=account_to.name, amount=amount)).id
  await db.query_single('update Account filter .name = <str>$name set {money := <int64>$amount, transactions += (select detached Transaction filter .id = <std::uuid>$id)}', name=account.name, id=t, amount=account.money-amount)
  await db.query_single('update Account filter .name = <str>$name set {money := <int64>$amount, transactions += (select detached Transaction filter .id = <std::uuid>$id)}', name=account_to.name, id=t, amount=account_to.money+amount)
  for i in wss.get(to, []):
    await i.send_json({"type": "receive", "id": str(t)})
  for i in wss.get(name, []):
    await i.send_json({"type": "send", "id": str(t)})
  return PlainTextResponse(str(t))

@app.websocket("/event")
async def event(ws: WebSocket):
  await ws.accept()
  try:
    data = await ws.receive_json()
  except JSONDecodeError:
    await ws.send_json({"status_code": 400, "detail": "Data is invalid"})
    await ws.close()
    return
  try:
    sha_hash = sha3_512(data['password'].encode('utf8') + salt).hexdigest()
    conv_to_dict(await db.query_single("select Account {name, money} filter .name = <str>$name and .password = <str>$password limit 1", name=data['name'], password=sha_hash))
  except HTTPException:
    await ws.send_json({"status_code": 400, "detail": "Login failed"})
    await ws.close()
    return
  if wss.get(data['name']) is None:
    wss[data['name']] = []
  wss[data['name']].append(ws)
  try:
    while True:
      data = await ws.receive_text()
      await ws.send_json({'ping': 'pong'})
  except WebSocketDisconnect:
    wss[data['name']].remove(ws)
    if len(wss[data['name']]) == 0:
      del wss[data['name']]
