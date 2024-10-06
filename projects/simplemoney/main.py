from fastapi import FastAPI, Header, status, HTTPException
from edgedb import create_async_client

db = create_async_client()
app = FastAPI()

class Transaction:
  amount: int
  date: str
  description: str

class Account[T]:
  money: int
  name: str
  password: str
  transactions: list[T]

class AccountPublic:
  money: int
  name: str
  transactions: list[str]

@app.get("/")
async def ping():
  return {"ping": "pong"}

@app.get("/user")
async def get_user(
  name: str | None = Header()
) -> AccountPublic:
  if not isinstance(name, str):
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
  user: Account[str] | None = await db.query_single("select Account {name, money, password, transactions: {id}} filter .name = <str>$name limit 1", name=name)
  if user is None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
  del user.password
  return user

@app.get("/transaction")
async def get_transaction(
  id: str | None = Header()
) -> Transaction:
  if not isinstance(id, str):
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
  transaction: Transaction | None = await db.query_single("select Transaction {amount, date, description} filter .id = <str>$id limit 1", id=id)
  if transaction is None:
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
  return transaction

@app.post("/account/send")
async def send_money(
  id: str | None = Header(),
  password: str | None = Header(),
  amount: int | None = Header()
):
  # check headers
  # just hash password
  # if not just gives error
  # check if money - amount > 0 and amount > 0
  # if not just gives error
  # query create transaction
  # return transaction id
  # profit
  pass