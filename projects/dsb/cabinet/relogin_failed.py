from lib import get_proxy, api
from asyncio import run

async def main():
  api.proxy = get_proxy()
  await api.pool.relogin_failed()

if __name__ == "__main__":
  run(main())
