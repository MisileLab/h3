from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import FastAPI, Header, status, HTTPException, WebSocket, WebSocketDisconnect, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import PlainTextResponse
from satellite_py import DB
from loguru import logger

from dataclasses import dataclass, field, asdict
from json.decoder import JSONDecodeError
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, TYPE_CHECKING, TypeVar
from uuid import uuid4, UUID
from contextlib import suppress

if TYPE_CHECKING:
  from _typeshed import DataclassInstance
else:
  DataclassInstance = TypeVar('DataclassInstance')

db = DB()
wss: defaultdict[str, list[WebSocket]] = defaultdict(list)
api_key = APIKeyHeader(name="x-api-key")
if not Path("./api_key").is_file():
  _ = Path("./api_key").write_text(uuid4().hex)
pw = PasswordHasher().hash(Path("./api_key").read_text().strip("\n"))
logger.debug("API key: {}", Path("./api_key").read_text())

def verify_uuid(v: str) -> bool:
  r: UUID | str = ""
  with suppress(ValueError):
    r = UUID(v)
  return str(r) == v

def get_api_key(api_key: Annotated[str, Security(api_key)]) -> str:
  global pw
  with suppress(VerifyMismatchError):
    a = PasswordHasher().verify(pw, api_key)
    if PasswordHasher().check_needs_rehash(pw):
      pw = PasswordHasher().hash(Path("./pw").read_text().strip("\n"))
    if a:
      return api_key
  raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API key is invalid")

app = FastAPI(dependencies=[Depends(get_api_key)])

@dataclass
class Account:
  money: int
  name: str
  password: str | None
  transactions: list[str] = field(default_factory=list)

@dataclass
class Transaction:
  amount: int
  to: str
  received: str

def conv_to_dict(
  v: DataclassInstance | None,
  status_code: int = 400,
  detail: str = "result of query is None (probably name or password is invalid)",
  ignore_id: bool = True
) -> dict[str, Any]: # pyright: ignore[reportExplicitAny]
  if v is None:
    raise HTTPException(status_code=status_code, detail=detail)
  res = asdict(v)
  if ignore_id:
    del res["id"]
  return res

def verify_pw(hash: str, v: str) -> bool:
  try:
    ph = PasswordHasher()
    _ = ph.verify(hash, v)
  except VerifyMismatchError as e:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="name or password is mismatched") from e
  return ph.check_needs_rehash(hash)

@app.get("/user")
async def get_user(
  name: Annotated[str, Header()]
) -> Account:
  _user = conv_to_dict(await db.query_single("""
    select Account {name, money, transactions: {id}}
    filter .name = <str>$name limit 1
  """,
  name=name), status_code=404, detail="user not found")
  _user['transactions'] = [str(i['id']) for i in _user['transactions']]
  return Account(password=None, **_user)

@app.get("/transaction")
async def get_transaction(
  id: Annotated[str, Header()]
) -> Transaction:
  if not verify_uuid(id):
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="id is invalid")
  return Transaction(**conv_to_dict(
    await db.query_single("select Transaction {amount, to, received} filter .id = <std::uuid>$id limit 1", id=id),
    status_code=404,
    detail="transaction not found"
  ))

@app.post("/account/send")
async def send_money(
  name: Annotated[str, Header()],
  password: Annotated[str, Header()],
  to: Annotated[str, Header()],
  amount: Annotated[int, Header()]
) -> PlainTextResponse:
  if name == to:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="can't send to same account")
  _account = conv_to_dict(await db.query_single("select Account {name, money, password} filter .name = <str>$name limit 1", name=name))
  if verify_pw(_account['password'], password):
    _ = await db.query_single("update Account filter .name = <str>$name set {password := <str>$password}", password=PasswordHasher().hash(password))
  _account_to = conv_to_dict(
    await db.query_single("select Account {name, money} filter .name = <str>$name limit 1", name=to),
    detail="destination is not found"
  )
  _account["password"] = None
  _account_to["password"] = None
  account = Account(**_account)
  account_to = Account(**_account_to)
  if account.money - amount < 0 or amount <= 0:
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="money amount is less than 0 or your money is not sufficient")
  t = conv_to_dict(await db.query_single("""insert Transaction {
    to := <str>$to,
    received := <str>$received,
    amount := <int64>$amount
  }""", to=account.name, received=account_to.name, amount=amount), ignore_id=False)['id']
  _ = await db.query_single("""update Account filter .name = <str>$name set {
    money := <int64>$amount,
    transactions += (select detached Transaction filter .id = <std::uuid>$id)
  }""", name=account.name, id=t, amount=account.money-amount)
  _ = await db.query_single("""update Account filter .name = <str>$name set {
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
      _ = await db.query_single("update Account filter .name = <str>$name {password := <str>$password}", password=PasswordHasher().hash(data['password']))
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
