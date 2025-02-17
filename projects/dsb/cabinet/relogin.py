from loguru import logger

from lib import get_usernames, api
from asyncio import run
from time import sleep
from secrets import SystemRandom

async def main():
  for i in await get_usernames():
    logger.info(f"Relogging {i}")
    await api.pool.relogin([i])
    sleep(SystemRandom().randint(1, 3))

if __name__ == "__main__":
  run(main())
