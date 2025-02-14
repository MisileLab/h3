from lib import api
from asyncio import run

async def main():
  await api.pool.relogin_failed()

if __name__ == "__main__":
  run(main())
