from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any

from disnake import ApplicationCommandInteraction, User
from disnake.ext.commands import Bot, Param, is_owner
from disnake.ext.tasks import loop
from edgedb import create_async_client  # pyright: ignore[reportUnknownVariableType]
from tomli import loads
from tomli_w import dumps

from queries.bank.get_bank_async_edgeql import get_bank
from queries.bank.is_bank_owner_async_edgeql import is_bank_owner
from queries.bank.modify_bank_async_edgeql import modify_bank
from queries.products.add_product_async_edgeql import add_product
from queries.products.delete_product_async_edgeql import delete_product
from queries.credit.get_credit_async_edgeql import get_credit
from queries.credit.set_credit_async_edgeql import set_credit
from queries.loan.get_loan_expired_async_edgeql import get_loan_expired
from queries.loan.refresh_loan_async_edgeql import refresh_loan
from queries.loan.reset_loan_async_edgeql import reset_loan
from queries.money.get_money_async_edgeql import get_money
from queries.money.set_money_async_edgeql import set_money
from queries.user.get_user_async_edgeql import get_user
from queries.user.is_any_bank_owner_async_edgeql import is_any_bank_owner

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

@loop(seconds=1)
async def loan_check():
  for i in await get_loan_expired(db):
    receiver = verify_none(await get_user(db, id=i.receiver))
    _ = await refresh_loan(
      db,
      date=i.date + timedelta(i.product.end_date),
      id=i.id,
      interest=int(i.interest + (i.amount / 100) * i.product.interest),
      credit=receiver.credit - 1,
      userid=i.receiver
    )
  pass

@bot.slash_command(
  name="지급",
  description="화폐를 지급하거나 제거함"
)
@is_owner()
async def give(inter: interType, user: User, amount: int):
  await inter.response.defer(ephemeral=True)
  money = verify_none(await get_money(db, userid=user.id))
  if money + amount < 0:
    await inter.send("지급 결과가 0 미만입니다.", ephemeral=True)
    return
  _ = await set_money(db, userid=user.id, money=money + amount)
  await inter.send("지급 완료", ephemeral=True)

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
  await inter.send("수수료 설정 완료", ephemeral=True)

@bot.slash_command(name="잔고", description="잔고를 확인함")
@is_owner()
async def storage(inter: interType, user: User):
  await inter.response.defer(ephemeral=True)
  money = verify_none(await get_money(db, userid=user.id))
  await inter.send(str(money), ephemeral=True)

@bot.slash_command(name="은행")
async def bank():
  pass

@bank.sub_command(name="설정", description="은행을 설정함 (없을 경우 추가, 이름은 변경 불가능함)") # pyright: ignore[reportUnknownMemberType]
@is_owner()
async def bank_setting(
  inter: interType,
  name: str,
  owner: User | None = None,
  money: int | None = None
):
  await inter.response.defer(ephemeral=True)
  bank = await get_bank(db, name=name)
  if bank is None:
    if owner is None or money is None:
      await inter.send("은행을 추가하려는 경우 모든 정보를 써야합니다.", ephemeral=True)
      return
    ownerid = owner.id
  else:
    ownerid = bank.owner.userid
    money = bank.money
  _ = await modify_bank(db, name=name, owner=ownerid, money=money)
  await inter.send("은행 변경 완료", ephemeral=True)

@bank.sub_command(name="대출생성", description="은행에서 대출 상품 생성함") # pyright: ignore[reportUnknownMemberType]
async def bank_loan_create(
  inter: interType,
  bank_name: str,
  name: str,
  min_trust: int,
  interest: Annotated[int, Param(ge=-1)],
  end_date: Annotated[int, Param(ge=0)]
):
  await inter.response.defer(ephemeral=True)
  if not verify_none(await is_bank_owner(db, name=bank_name, ownerid=inter.author.id)):
    await inter.send("은행의 소유자가 아닙니다.", ephemeral=True)
    return
  _ = await add_product(db, bank_name=bank_name, name=name, min_trust=min_trust, end_date=end_date, interest=interest)
  await inter.send("상품 생성 완료", ephemeral=True)

@bank.sub_command(name="대출삭제", description="은행에서 대출 상품 삭제함") # pyright: ignore[reportUnknownMemberType]
async def bank_loan_delete(inter: interType, bank_name: str, name: str):
  await inter.response.defer(ephemeral=True)
  if not verify_none(await is_bank_owner(db, name=bank_name, ownerid=inter.author.id)):
    await inter.send("은행의 소유자가 아닙니다.", ephemeral=True)
    return
  _ = delete_product(db, bank_name=bank_name, name=name)
  await inter.send("상품 삭제 완료", ephemeral=True)

@bank.sub_command(name="신용도확인", description="유저의 신용도 확인") # pyright: ignore[reportUnknownMemberType]
async def bank_credit(inter: interType, user: User):
  await inter.response.defer(ephemeral=True)
  if not verify_none(await is_any_bank_owner(db, userid=inter.author.id)):
    await inter.send("은행의 소유자가 아닙니다.", ephemeral=True)
    return
  await inter.send(str(verify_none(await get_credit(db, userid=user.id))), ephemeral=True)

@bot.slash_command(name="신용도")
async def credit():
  pass

@credit.sub_command(name="설정", description="신용도를 설정함") # pyright: ignore[reportUnknownMemberType]
@is_owner()
async def credit_setting(inter: interType, user: User, credit: int):
  await inter.response.defer(ephemeral=True)
  _ = await set_credit(db, userid=user.id, credit=credit)
  await inter.send("신용도 설정 완료", ephemeral=True)

@bot.slash_command(name="대출")
async def loan():
  pass

@loan.sub_command(name="리셋", description="대출 기록을 리셋함") # pyright: ignore[reportUnknownMemberType]
@is_owner()
async def loan_reset(inter: interType, user: User):
  await inter.response.defer(ephemeral=True)
  _ = await reset_loan(db, userid=user.id)
  await inter.send("대출 기록 리셋 완료", ephemeral=True)

bot.run(config["token"])

