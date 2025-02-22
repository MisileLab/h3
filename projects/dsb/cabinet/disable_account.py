from lib import api
from asyncio import run

async def main():
  acc_name = input().strip()
  await api.pool.set_active(acc_name, False)

if __name__ == "__main__":
  run(main())

