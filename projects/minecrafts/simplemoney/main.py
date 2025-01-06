from pathlib import Path
from typing import Any

from disnake import ApplicationCommandInteraction, User
from disnake.ext.commands import Bot, is_owner
from edgedb import create_async_client  # pyright: ignore[reportUnknownVariableType]
from tomli import loads

from queries.money.get_money_async_edgeql import get_money
from queries.money.set_money_async_edgeql import set_money

type interType = ApplicationCommandInteraction[Any] # pyright: ignore[reportExplicitAny]

db = create_async_client()
bot = Bot(
  help_command=None,
  owner_ids={735677489958879324, 338902243476635650},
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

bot.run(config["token"])

