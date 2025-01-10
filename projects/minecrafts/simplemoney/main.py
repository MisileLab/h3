from contextlib import suppress
from datetime import UTC, timedelta
from math import ceil
from pathlib import Path
from traceback import print_exception
from typing import Any, final
from datetime import datetime

from disnake import ApplicationCommandInteraction, User
from disnake.errors import InteractionResponded
from disnake.ext.commands import CommandInvokeError, InteractionBot, MissingPermissions, Param, has_guild_permissions
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
from queries.bank.get_bank_products_async_edgeql import get_bank_products
from queries.products.get_product_async_edgeql import get_product
from queries.products.add_product_async_edgeql import add_product
from queries.products.delete_product_async_edgeql import delete_product
from queries.credit.get_credit_async_edgeql import get_credit
from queries.credit.set_credit_async_edgeql import set_credit
from queries.loan.get_loan_expired_async_edgeql import get_loan_expired
from queries.loan.refresh_loan_async_edgeql import refresh_loan
from queries.loan.reset_loan_async_edgeql import reset_loan
from queries.loan.get_loan_bank_async_edgeql import get_loan_bank
from queries.loan.get_loan_user_async_edgeql import get_loan_user
from queries.loan.get_loan_amount_async_edgeql import get_loan_amount
from queries.loan.add_loan_async_edgeql import add_loan
from queries.loan.pay_loan_async_edgeql import pay_loan
from queries.money.set_money_async_edgeql import set_money
from queries.money.get_money_async_edgeql import get_money
from queries.user.get_user_banks_async_edgeql import get_user_banks
from queries.user.is_any_bank_owner_async_edgeql import is_any_bank_owner
from queries.user.deposit_to_bank_async_edgeql import deposit_to_bank
from queries.user.send_async_edgeql import send
from queries.user.get_user_by_uuid_async_edgeql import get_user_by_uuid

type interType = ApplicationCommandInteraction[Any] # pyright: ignore[reportExplicitAny]

class sendFee(BaseModel):
  send: int

class subFee(sendFee):
  receive: int

class Fee(subFee):
  borrow: subFee
  bank: sendFee 

class Config(BaseModel):
  fees: Fee
  token: str

db = create_async_client()
bot = InteractionBot(
  owner_ids={735677489958879324},
  test_guilds=[1322924155640877056]
)
config = Config.model_validate(loads(Path("./config_prod.toml").read_text()))

@final
class ValueNoneError(Exception):
  def __init__(self, message: str):
    super().__init__(message)
    self.message = message

@bot.event
async def on_ready():
  print("ready")

@bot.event
async def on_slash_command_error(ctx: interType, error: CommandInvokeError):
  with suppress(InteractionResponded):
    await ctx.response.defer(ephemeral=True)
  if isinstance(error, MissingPermissions):
    _ = await ctx.edit_original_message("you don't have permission")
    return
  err = error.original
  if isinstance(err, ValueNoneError) and err.message != "it's None":
    _ = await ctx.edit_original_message(err.message)
    return
  print_exception(error)
  _ = await ctx.edit_original_message("error (probably you are wrong)")

def verify_none[T](v: T | None, message: str = "it's None") -> T:
  if v is None:
    raise ValueNoneError(message)
  return v

@loop(seconds=1)
async def loan_refresh():
  for i in await get_loan_expired(db):
    receiver = verify_none(await get_user_by_uuid(db, id=i.receiver))
    _ = await refresh_loan(
      db,
      date=i.date + timedelta(i.product.end_date),
      id=i.id,
      amount=i.amount + ceil((i.amount / 100) * i.product.interest),
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
    bank = verify_none(await get_bank_by_id(db, id=i.receiver))
    contents.append(f"{bank.name}: {i.amount}")
  _ = await inter.edit_original_message(f"현금: {money.money}\n{'\n'.join(contents)}")

@bot.slash_command(name="송금", description="돈을 다른 유저에게 송금함")
async def send_money(inter: interType, user: User, amount: int):
  await inter.response.defer(ephemeral=True)
  sender_money = verify_none(await get_money(db, userid=inter.author.id))
  receiver_money = verify_none(await get_money(db, userid=user.id))
  _ = await send(
    db,
    receiverid=user.id,
    senderid=inter.author.id,
    amount=amount,
    sender_money=sender_money - amount,
    receiver_money=receiver_money + amount - ceil(amount / 100 * config.fees.send)
  )
  _ = await inter.edit_original_message("송금 완료")

@bot.slash_command(
  name="지급",
  description="화폐를 지급하거나 제거함"
)
@has_guild_permissions(administrator=True)
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
@has_guild_permissions(administrator=True)
async def fee(
  inter: interType,
  send_fee: int = Param(
    ge=-1,
    le=100,
    description="송금 수수료",
    default=config.fees.send 
  ), # pyright: ignore[reportCallInDefaultInitializer]
  receive_fee: int = Param(
    ge=-1,
    le=100,
    description="출금 수수료",
    default=config.fees.receive
  ), # pyright: ignore[reportCallInDefaultInitializer]
  borrow_send_fee: int = Param(
    ge=-1,
    le=100,
    description="대출 갚는 수수료",
    default=config.fees.borrow.send
  ), # pyright: ignore[reportCallInDefaultInitializer]
  borrow_receive_fee: int = Param(
    ge=-1,
    le=100,
    description="대출 수수료",
    default=config.fees.borrow.receive
  ), # pyright: ignore[reportCallInDefaultInitializer]
  bank_send_fee: int = Param(
    ge=-1,
    le=100,
    description="은행끼리의 송금 수수료",
    default=config.fees.bank.send
  ) # pyright: ignore[reportCallInDefaultInitializer]
):
  await inter.response.defer(ephemeral=True)
  config.fees.send = send_fee
  config.fees.receive = receive_fee
  config.fees.borrow.send = borrow_send_fee
  config.fees.borrow.receive = borrow_receive_fee
  config.fees.bank.send = bank_send_fee
  _ = Path("./config_prod.toml").write_text(dumps(config.model_dump()))
  _ = await inter.edit_original_message("수수료 설정 완료")

@bot.slash_command(name="잔고", description="잔고를 확인함")
@has_guild_permissions(administrator=True)
async def storage(inter: interType, user: User):
  await inter.response.defer(ephemeral=True)
  money = verify_none(await get_user_banks(db, userid=user.id))
  contents: list[str] = []
  for i in money.banks:
    bank = verify_none(await get_bank_by_id(db, id=i.receiver))
    contents.append(f"{bank.name}: {i.amount}")
  _ = await inter.edit_original_message(f"현금: {money.money}\n{'\n'.join(contents)}")

@bot.slash_command(name="은행")
async def bank(_: interType):
  pass

@bank.sub_command(name="잔고", description="은행의 잔고를 확인함") # pyright: ignore[reportUnknownMemberType]
async def bank_balance(inter: interType, bank_name: str):
  await inter.response.defer(ephemeral=True)
  if not verify_none(await is_bank_owner(db, name=bank_name, ownerid=inter.author.id)):
    _ = await inter.edit_original_message("은행의 소유자가 아닙니다.")
    return
  bank = verify_none(await get_bank(db, name=bank_name))
  _ = await inter.edit_original_message(str(bank.money))

@bank.sub_command(name="입금", description="은행에 돈을 입금함") # pyright: ignore[reportUnknownMemberType]
async def bank_deposit(inter: interType, bank_name: str, amount: int):
  await inter.response.defer(ephemeral=True)
  bank = verify_none(await get_bank(db, name=bank_name))
  _ = await deposit_to_bank(
    db,
    receiver=bank.id,
    senderid=inter.author.id,
    amount=amount
  )
  _ = await inter.edit_original_message("입금 완료")

@bank.sub_command(name="출금", description="은행에서 돈을 출금함") # pyright: ignore[reportUnknownMemberType]
async def bank_withdraw(inter: interType, bank_name: str, amount: int):
  await inter.response.defer(ephemeral=True)
  bank = verify_none(
    await get_bank(db, name=bank_name),
    "은행이 존재하지 않습니다."
  )
  money = verify_none(await get_money(db, userid=inter.author.id))
  _ = await send_to_user(
    db,
    receiverid=inter.author.id,
    sender=bank.id,
    amount=amount,
    sender_money=bank.money - amount,
    receiver_money=money + amount - ceil(amount / 100 * config.fees.receive)
  )
  _ = await inter.edit_original_message("출금 완료")

@bank.sub_command(name="설정", description="은행을 설정함 (없을 경우 추가, 이름은 변경 불가능함)") # pyright: ignore[reportUnknownMemberType]
@has_guild_permissions(administrator=True)
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
  min_trust: int = Param(description="최소 신용도"), # pyright: ignore[reportCallInDefaultInitializer]
  interest: int = Param(ge=-1, description="이자, 단위: 퍼센트"), # pyright: ignore[reportCallInDefaultInitializer]
  end_date: int = Param(ge=0, description="단위: 일"), # pyright: ignore[reportCallInDefaultInitializer]
  max_amount: int = Param(ge=0, description="최대 대출 가능 금액") # pyright: ignore[reportCallInDefaultInitializer]
):
  await inter.response.defer(ephemeral=True)
  if not verify_none(await is_bank_owner(db, name=bank_name, ownerid=inter.author.id)):
    _ = await inter.edit_original_message("은행의 소유자가 아닙니다.")
    return
  _ = await add_product(
    db,
    bank_name=bank_name,
    name=name,
    min_trust=min_trust,
    end_date=end_date,
    interest=interest,
    max_amount=max_amount
  )
  _ = await inter.edit_original_message("상품 생성 완료")

@bank.sub_command(name="대출삭제", description="은행에서 대출 상품 삭제함") # pyright: ignore[reportUnknownMemberType]
async def bank_loan_delete(inter: interType, bank_name: str, name: str):
  await inter.response.defer(ephemeral=True)
  if not verify_none(await is_bank_owner(db, name=bank_name, ownerid=inter.author.id)):
    _ = await inter.edit_original_message("은행의 소유자가 아닙니다.")
    return
  _ = await delete_product(db, bank_name=bank_name, name=name)
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
        i.product.name,
        f"<t:{int(i.date.timestamp())}:R>",
        str(i.amount),
        f"{i.product.interest}%",
        f"<@{verify_none(await get_user_by_uuid(db, id=i.receiver)).userid}>"
      ])
    )
  _ = await inter.edit_original_message(content=f"상품 이름,만기일,돈,이자율,유저\n{'\n'.join(contents)}")

@bank.sub_command(name="송금", description="은행 돈을 다른 곳으로 송금") # pyright: ignore[reportUnknownMemberType]
async def bank_send(
    inter: interType,
    bank_name: str,
    amount: int,
    destination_user: User | None = Param(description="송금할 유저(은행이나 유저 둘 중 하나만 설정해야함)"), # pyright: ignore[reportCallInDefaultInitializer]
    destination_bank: str | None = Param(description="송금할 은행(은행이나 유저 둘 중 하나만 설정해야함)") # pyright: ignore[reportCallInDefaultInitializer]
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
      ) + amount - ceil(amount / 100 * config.fees.bank.send)
    )
  elif destination_bank is not None:
    receiver = verify_none(await get_bank(db, name=destination_bank))
    _ = await send_to_bank(
      db,
      receiver=receiver.id,
      sender=sender.id,
      amount=amount,
      sender_money=sender.money - amount,
      receiver_money=receiver.money + amount - ceil(amount / 100 * config.fees.bank.send)
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
async def credit(_: interType):
  pass

@credit.sub_command(name="설정", description="신용도를 설정함") # pyright: ignore[reportUnknownMemberType]
@has_guild_permissions(administrator=True)
async def credit_setting(inter: interType, user: User, credit: int):
  await inter.response.defer(ephemeral=True)
  _ = await set_credit(db, userid=user.id, credit=credit)
  _ = await inter.edit_original_message("신용도 설정 완료")

@credit.sub_command(name="확인", description="신용도를 확인함") # pyright: ignore[reportUnknownMemberType]
async def credit_check(inter: interType):
  await inter.response.defer(ephemeral=True)
  _ = await inter.edit_original_message(str(verify_none(await get_credit(db, userid=inter.author.id))))

@bot.slash_command(name="대출")
async def loan(_: interType):
  pass

@loan.sub_command(name="상품목록", description="대출 상품 목록을 확인함") # pyright: ignore[reportUnknownMemberType]
async def loan_product(inter: interType, bank_name: str):
  await inter.response.defer(ephemeral=True)
  contents: list[str] = []
  for i in verify_none(await get_bank_products(db, name=bank_name)).products:
    contents.append(
      f"{i.name},{i.min_trust},{i.interest}%,{i.end_date}일,{i.max_amount}"
    )
  _ = await inter.edit_original_message(content=f"상품 이름,신용도,이자율,만기일,최대 대출가능한 돈\n{'\n'.join(contents)}")

@loan.sub_command(name="리셋", description="대출 기록을 리셋함") # pyright: ignore[reportUnknownMemberType]
@has_guild_permissions(administrator=True)
async def loan_reset(inter: interType, user: User):
  await inter.response.defer(ephemeral=True)
  _ = await reset_loan(db, userid=user.id)
  _ = await inter.edit_original_message("대출 기록 리셋 완료")

@loan.sub_command(name="확인", description="현재 대출 목록을 확인함") # pyright: ignore[reportUnknownMemberType]
async def loan_check(inter: interType):
  await inter.response.defer(ephemeral=True)
  contents: list[str] = []
  for i in verify_none(await get_loan_user(db, userid=inter.author.id)):
    contents.append(
      f"{i.product.name},{i.amount}원"
    )
  if len(contents) == 0:
    _ = await inter.edit_original_message("대출한 것이 없습니다.")
  _ = await inter.edit_original_message(content=f"상품 이름,갚을 돈\n{'\n'.join(contents)}")

@loan.sub_command(name="시작", description="대출을 시작함") # pyright: ignore[reportUnknownMemberType]
async def loan_start(inter: interType, product_name: str, bank_name: str, amount: int):
  await inter.response.defer(ephemeral=True)
  product = verify_none(
    await get_product(db, bank_name=bank_name, name=product_name),
    "그런 상품이 없습니다."
  )
  user = verify_none(await get_user_banks(db, userid=inter.author.id))
  res = await get_loan_amount(
      db,
      userid=inter.author.id,
      sender=verify_none(await get_bank(db, name=bank_name)).id,
      product_name=product_name
    )
  loan = 0 if res is None else res.amount
  if user.credit < product.min_trust:
    _ = await inter.edit_original_message("신용도가 부족합니다.")
    return
  if amount + loan > product.max_amount:
    _ = await inter.edit_original_message("최대 대출 가능 금액을 초과했습니다.")
    return
  _ = await add_loan(
    db,
    product_id=product.id,
    bank_name=bank_name,
    bank_money=verify_none(await get_bank(db, name=bank_name)).money - amount,
    amount=amount + ceil(amount / 100 * product.interest),
    receiver_id=inter.author.id,
    receiver_money=verify_none(await get_money(db, userid=inter.author.id)) + amount - ceil(amount / 100 * config.fees.borrow.send),
    date=datetime.now(UTC) + timedelta(product.end_date)
  )
  _ = await inter.edit_original_message("대출 완료")

@loan.sub_command(name="갚기", description="대출을 갚음") # pyright: ignore[reportUnknownMemberType]
async def loan_pay(inter: interType, product_name: str, bank_name: str, amount: int):
  await inter.response.defer(ephemeral=True)
  amount -= ceil(amount / 100 * config.fees.borrow.receive)
  _ = await pay_loan(
    db,
    bank_name=bank_name,
    product_name=product_name,
    amount=amount,
    receiver_id=inter.author.id,
    receiver_money=verify_none(await get_money(db, userid=inter.author.id)) + amount,
    bank_money=verify_none(await get_bank(db, name=bank_name)).money + amount
  )
  _ = await inter.edit_original_message("대출 갚기 완료")

bot.run(config.token)

