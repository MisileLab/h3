from twscrape import API # pyright: ignore[reportMissingTypeStubs]

from lib import get_proxy, get_usernames
from asyncio import run

async def main():
  api = API()
  api.proxy = get_proxy()
  await api.pool.relogin(await get_usernames())

if __name__ == "__main__":
  run(main())
