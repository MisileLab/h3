from lib import get_usernames, api
from asyncio import run

async def main():
  await api.pool.relogin(await get_usernames())

if __name__ == "__main__":
  run(main())
