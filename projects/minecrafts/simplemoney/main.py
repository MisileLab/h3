from pathlib import Path
from typing import Any

from disnake import ApplicationCommandInteraction, User
from disnake.ext.commands import Bot, is_owner
from edgedb import create_async_client  # pyright: ignore[reportUnknownVariableType]
from tomli import loads

from queries.money.get_money_async_edgeql import get_money
from queries.money.set_money_async_edgeql import set_money
from queries.bank.modify_bank_async_edgeql import modify_bank
from queries.bank.get_bank_async_edgeql import get_bank

type interType = ApplicationCommandInteraction[Any] # pyright: ignore[reportExplicitAny]

db = create_async_client()
bot = Bot(
  help_command=None,
  owner_ids={338902243476635650},
  test_guilds=[1322924155640877056]
)
config = loads(Path("./config.toml").read_text())

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
  credit = verify_none(await get_money(db, userid=user.id))
  if credit + amount < 0:
    await inter.send("지급 결과가 0 미만입니다.")
    return
  _ = await set_money(db, userid=user.id, credit=credit + amount)
  await inter.send("지급 완료")

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

bot.run(config["token"])

