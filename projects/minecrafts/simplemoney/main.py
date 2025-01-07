from pathlib import Path
from typing import Any, Annotated

from disnake import ApplicationCommandInteraction, User
from disnake.ext.commands import Bot, Param, is_owner
from edgedb import create_async_client  # pyright: ignore[reportUnknownVariableType]
from tomli import loads
from tomli_w import dumps

from queries.money.get_money_async_edgeql import get_money
from queries.money.set_money_async_edgeql import set_money
from queries.bank.modify_bank_async_edgeql import modify_bank
from queries.bank.get_bank_async_edgeql import get_bank
from queries.credit.set_credit_async_edgeql import set_credit
from queries.loan.reset_loan_async_edgeql import reset_loan

type interType = ApplicationCommandInteraction[Any] # pyright: ignore[reportExplicitAny]

db = create_async_client()
bot = Bot(
  help_command=None,
  owner_ids={338902243476635650},
  test_guilds=[1322924155640877056]
)
config = loads(Path("./config_prod.toml").read_text())

def verify_none[T](v: T | None) -> T:
  if v is None:
    raise TypeError("it's None")
  return v

@bot.slash_command(
  name="지급",
  description="화폐를 지급하거나 제거함"
)
@is_owner()
async def give(inter: interType, user: User, amount: int):
  await inter.response.defer()
  money = verify_none(await get_money(db, userid=user.id))
  if money + amount < 0:
    await inter.send("지급 결과가 0 미만입니다.")
    return
  _ = await set_money(db, userid=user.id, money=money + amount)
  await inter.send("지급 완료")

@bot.slash_command(
  name="수수료",
  description="수수료를 설정함"
)
@is_owner()
async def fee(
  inter: interType,
  send_fee: Annotated[int, Param(
    ge=-1,
    description="송금 수수료",
    default=config["fee"]["send"] # pyright: ignore[reportAny]
  )],
  receive_fee: Annotated[int, Param(
    ge=-1,
    description="출금 수수료",
    default=config["fee"]["receive"] # pyright: ignore[reportAny]
  )],
  borrow_send_fee: Annotated[int, Param(
    ge=-1,
    description="대출 갚는 수수료",
    default=config["fee"]["borrow"]["send"] # pyright: ignore[reportAny]
  )],
  borrow_receive_fee: Annotated[int, Param(
    ge=-1,
    description="대출 수수료",
    default=config["fee"]["borrow"]["receive"] # pyright: ignore[reportAny]
  )],
  bank_send_fee: Annotated[int, Param(
    ge=-1,
    description="은행끼리의 송금 수수료",
    default=config["fee"]["bank"]["send"] # pyright: ignore[reportAny]
  )]
):
  await inter.response.defer()
  config["fee"]["send"] = send_fee
  config["fee"]["receive"] = receive_fee
  config["fee"]["borrow"]["send"] = borrow_send_fee
  config["fee"]["borrow"]["receive"] = borrow_receive_fee
  config["fee"]["bank"]["send"] = bank_send_fee
  _ = Path("./config_prod.toml").write_text(dumps(config))
  await inter.send("수수료 설정 완료")

@bot.slash_command(name="잔고", description="잔고를 확인함")
@is_owner()
async def storage(inter: interType, user: User):
  await inter.response.defer()
  money = verify_none(await get_money(db, userid=user.id))
  await inter.send(str(money))

@bot.slash_command(name="은행")
async def bank():
  pass

@bank.sub_command(name="설정", description="은행을 설정함 (없을 경우 추가, 이름은 변경 불가능함)") # pyright: ignore[reportUnknownMemberType]
@is_owner()
async def bank_setting(
  inter: interType,
  name: str,
  owner: User | None = None,
  amount: int | None = None
):
  await inter.response.defer()
  bank = await get_bank(db, name=name)
  if bank is None:
    if None in [amount, bank]:
      await inter.send("은행을 추가하려는 경우 모든 정보를 써야합니다.")
      return
    raise ValueError("Unreachable")
  ownerid = owner.id if owner is not None else bank.owner.userid
  amount = amount if amount is not None else bank.amount
  _ = await modify_bank(db, name=name, owner=ownerid, amount=amount)

@bot.slash_command(name="신용도")
async def credit():
  pass

@credit.sub_command(name="설정", description="신용도를 설정함") # pyright: ignore[reportUnknownMemberType]
@is_owner()
async def credit_setting(inter: interType, user: User, credit: int):
  await inter.response.defer()
  _ = await set_credit(db, userid=user.id, credit=credit)
  await inter.send("신용도 설정 완료")

@bot.slash_command(name="대출")
async def loan():
  pass

@loan.sub_command(name="리셋", description="대출 기록을 리셋함") # pyright: ignore[reportUnknownMemberType]
@is_owner()
async def loan_reset(inter: interType, user: User):
  await inter.response.defer()
  _ = await reset_loan(db, userid=user.id)
  await inter.send("대출 기록 리셋 완료")

bot.run(config["token"])

