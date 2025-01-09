from datetime import timedelta
from pathlib import Path
from traceback import print_exception
from typing import Annotated, Any

from disnake import ApplicationCommandInteraction, User
from disnake.ext.commands import Bot, Param, is_owner
from disnake.ext.tasks import loop
from edgedb import create_async_client  # pyright: ignore[reportUnknownVariableType]
from tomli import loads
from tomli_w import dumps
from pydantic import BaseModel

from queries.bank.get_bank_async_edgeql import get_bank
from queries.bank.get_bank_by_id_async_edgeql import get_bank_by_id
from queries.bank.is_bank_owner_async_edgeql import is_bank_owner
from queries.bank.modify_bank_async_edgeql import modify_bank
from queries.bank.send_to_user_async_edgeql import send_to_user
from queries.bank.send_to_bank_async_edgeql import send_to_bank
from queries.products.add_product_async_edgeql import add_product
from queries.products.delete_product_async_edgeql import delete_product
from queries.credit.get_credit_async_edgeql import get_credit
from queries.credit.set_credit_async_edgeql import set_credit
from queries.loan.get_loan_expired_async_edgeql import get_loan_expired
from queries.loan.refresh_loan_async_edgeql import refresh_loan
from queries.loan.reset_loan_async_edgeql import reset_loan
from queries.loan.get_loan_bank_async_edgeql import get_loan_bank
from queries.money.set_money_async_edgeql import set_money
from queries.money.get_money_async_edgeql import get_money
from queries.user.get_user_async_edgeql import get_user
from queries.user.get_user_banks_async_edgeql import get_user_banks
from queries.user.is_any_bank_owner_async_edgeql import is_any_bank_owner

type interType = ApplicationCommandInteraction[Any] # pyright: ignore[reportExplicitAny]

class sendFee(BaseModel):
  send: int

class subFee(sendFee):
  receive: int

class Fee(subFee):
  borrow: subFee
  bank: sendFee 

class Config(BaseModel):
  fee: Fee
  token: str

db = create_async_client()
bot = Bot(
  help_command=None,
  owner_ids={338902243476635650},
  test_guilds=[1322924155640877056]
)
config = Config.model_validate(loads(Path("./config_prod.toml").read_text()))

@bot.event
async def on_ready():
  print("ready")

@bot.event
async def on_command_error(ctx: interType, error: Exception):
  print_exception(error)
  await ctx.send("error (probably you are wrong)")

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

@bot.slash_command(name="보유량", description="보유하고 있는 은행과 현금 돈 확인")
async def money(inter: interType):
  await inter.response.defer(ephemeral=True)
  money = verify_none(await get_user_banks(db, userid=inter.author.id))
  contents: list[str] = []
  for i in money.banks:
    bank = verify_none(await get_bank_by_id(db, id=i.sender))
    contents.append(f"{bank.name}: {i.amount}")
  _ = await inter.edit_original_message(f"현금: {money.money}\n{'\n'.join(contents)}")

@bot.slash_command(
  name="지급",
  description="화폐를 지급하거나 제거함"
)
@is_owner()
async def give(inter: interType, user: User, amount: int):
  await inter.response.defer(ephemeral=True)
  money = verify_none(await get_money(db, userid=user.id))
  if money + amount < 0:
    _ = await inter.edit_original_message("지급 결과가 0 미만입니다.")
    return
  _ = await set_money(db, userid=user.id, money=money + amount)
  _ = await inter.edit_original_message("지급 완료")

@bot.slash_command(
  name="수수료",
  description="수수료를 설정함"
)
@is_owner()
async def fee(
  inter: interType,
  send_fee: Annotated[int, Param(
    ge=-1,
    le=100,
    description="송금 수수료",
    default=config.fee.send 
  )],
  receive_fee: Annotated[int, Param(
    ge=-1,
    le=100,
    description="출금 수수료",
    default=config.fee.receive
  )],
  borrow_send_fee: Annotated[int, Param(
    ge=-1,
    le=100,
    description="대출 갚는 수수료",
    default=config.fee.borrow.send
  )],
  borrow_receive_fee: Annotated[int, Param(
    ge=-1,
    le=100,
    description="대출 수수료",
    default=config.fee.borrow.receive
  )],
  bank_send_fee: Annotated[int, Param(
    ge=-1,
    le=100,
    description="은행끼리의 송금 수수료",
    default=config.fee.bank.send
  )]
):
  await inter.response.defer(ephemeral=True)
  config.fee.send = send_fee
  config.fee.receive = receive_fee
  config.fee.borrow.send = borrow_send_fee
  config.fee.borrow.receive = borrow_receive_fee
  config.fee.bank.send = bank_send_fee
  _ = Path("./config_prod.toml").write_text(dumps(config.model_dump()))
  _ = await inter.edit_original_message("수수료 설정 완료")

@bot.slash_command(name="잔고", description="잔고를 확인함")
@is_owner()
async def storage(inter: interType, user: User):
  await inter.response.defer(ephemeral=True)
  money = verify_none(await get_user_banks(db, userid=user.id))
  contents: list[str] = []
  for i in money.banks:
    bank = verify_none(await get_bank_by_id(db, id=i.sender))
    contents.append(f"{bank.name}: {i.amount}")
  _ = await inter.edit_original_message(f"현금: {money.money}\n{'\n'.join(contents)}")

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
      _ = await inter.edit_original_message("은행을 추가하려는 경우 모든 정보를 써야합니다.")
      return
    ownerid = owner.id
  else:
    ownerid = bank.owner.userid
    money = bank.money
  _ = await modify_bank(db, name=name, owner=ownerid, money=money)
  _ = await inter.edit_original_message("은행 변경 완료")

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
    _ = await inter.edit_original_message("은행의 소유자가 아닙니다.")
    return
  _ = await add_product(db, bank_name=bank_name, name=name, min_trust=min_trust, end_date=end_date, interest=interest)
  _ = await inter.edit_original_message("상품 생성 완료")

@bank.sub_command(name="대출삭제", description="은행에서 대출 상품 삭제함") # pyright: ignore[reportUnknownMemberType]
async def bank_loan_delete(inter: interType, bank_name: str, name: str):
  await inter.response.defer(ephemeral=True)
  if not verify_none(await is_bank_owner(db, name=bank_name, ownerid=inter.author.id)):
    _ = await inter.edit_original_message("은행의 소유자가 아닙니다.")
    return
  _ = delete_product(db, bank_name=bank_name, name=name)
  _ = await inter.edit_original_message("상품 삭제 완료")

@bank.sub_command(name="대출확인", description="진행중인 대출 확인") # pyright: ignore[reportUnknownMemberType]
async def bank_loan_check(inter: interType, bank_name: str):
  await inter.response.defer(ephemeral=True)
  if not verify_none(await is_bank_owner(db, name=bank_name, ownerid=inter.author.id)):
    _ = await inter.edit_original_message("은행의 소유자가 아닙니다.")
    return
  contents: list[str] = []
  for i in verify_none(await get_loan_bank(db, name=bank_name)):
    contents.append(
      ",".join([
        f"<t:{int(i.date.timestamp())}:R>",
        str(i.amount + i.interest),
        f"{i.product.interest}%",
        f"<@{verify_none(await get_user(db, id=i.receiver)).userid}>"
      ])
    )
  _ = await inter.edit_original_message(content=f"만기일,돈,이자율,유저\n{'\n'.join(contents)}")

@bank.sub_command(name="송금", description="은행 돈을 다른 곳으로 송금") # pyright: ignore[reportUnknownMemberType]
async def bank_send(
    inter: interType,
    bank_name: str,
    amount: Annotated[int, Param(ge=100)],
    destination_user: Annotated[User | None, Param(description="송금할 유저(은행이나 유저 둘 중 하나만 설정해야함)")],
    destination_bank: Annotated[str | None, Param(description="송금할 은행(은행이나 유저 둘 중 하나만 설정해야함)")]
  ):
  await inter.response.defer(ephemeral=True)
  if not verify_none(await is_bank_owner(db, name=bank_name, ownerid=inter.author.id)):
    _ = await inter.edit_original_message("은행의 소유자가 아닙니다.")
    return
  if [destination_bank, destination_user].count(None) != 1:
    _ = await inter.edit_original_message("도착지가 한 개가 아닙니다.")
    return
  sender = verify_none(await get_bank(db, name=bank.name))
  if destination_user is not None:
    _ = await send_to_user(
      db,
      receiverid=destination_user.id,
      sender=sender.id,
      amount=amount,
      sender_money=sender.money - amount,
      receiver_money=verify_none(
        await get_money(db, userid=destination_user.id)
      ) + amount - int(amount / 100 * config.fee.bank.send)
    )
  elif destination_bank is not None:
    receiver = verify_none(await get_bank(db, name=destination_bank))
    _ = await send_to_bank(
      db,
      receiver=receiver.id,
      sender=sender.id,
      amount=amount,
      sender_money=sender.money - amount,
      receiver_money=receiver.money + amount - int(amount / 100 * config.fee.bank.send)
    )
  else:
    raise ValueError("Unreachable")
  _ = await inter.edit_original_message("성공적으로 송금되었습니다.")

@bank.sub_command(name="신용도확인", description="유저의 신용도 확인") # pyright: ignore[reportUnknownMemberType]
async def bank_credit(inter: interType, user: User):
  await inter.response.defer(ephemeral=True)
  if not verify_none(await is_any_bank_owner(db, userid=inter.author.id)):
    _ = await inter.edit_original_message("은행의 소유자가 아닙니다.")
    return
  _ = await inter.edit_original_message(str(verify_none(await get_credit(db, userid=user.id))))

@bot.slash_command(name="신용도")
async def credit():
  pass

@credit.sub_command(name="설정", description="신용도를 설정함") # pyright: ignore[reportUnknownMemberType]
@is_owner()
async def credit_setting(inter: interType, user: User, credit: int):
  await inter.response.defer(ephemeral=True)
  _ = await set_credit(db, userid=user.id, credit=credit)
  _ = await inter.edit_original_message("신용도 설정 완료")

@bot.slash_command(name="대출")
async def loan():
  pass

@loan.sub_command(name="리셋", description="대출 기록을 리셋함") # pyright: ignore[reportUnknownMemberType]
@is_owner()
async def loan_reset(inter: interType, user: User):
  await inter.response.defer(ephemeral=True)
  _ = await reset_loan(db, userid=user.id)
  _ = await inter.edit_original_message("대출 기록 리셋 완료")

bot.run(config.token)

